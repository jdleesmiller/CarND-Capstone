import rospy

from twiddler import Twiddler

DEFAULT_DELTA_FACTOR = 0.2

#
# We want to start a new epoch when the controller crosses the minimum speed
# threshold from below. However, we don't want to do so immediately after a
# reset, because the car will ordinarily be stationary after a reset (or when
# first starting out), so we have a 'start' state that causes us to ignore the
# first crossing.
#
STATE_START = 0
STATE_OVER = 1
STATE_UNDER = 2


class SpeedPIDTuner(object):
    """
    Tune a PID controller for speed by measuring total absolute error between
    stops and adjusting the gains using the twiddle algorithm from lecture.
    """
    def __init__(self, pid, deltas=None, min_target=0):
        self.pid = pid
        self.min_target = min_target
        gains = [pid.kp, pid.ki, pid.kd]
        if deltas is None:
            deltas = [k * DEFAULT_DELTA_FACTOR for k in gains]
        self.twiddler = Twiddler(gains, deltas)
        self.crossing_state = STATE_START
        self.__start_epoch()

    def reset(self):
        """
        If we get reset, the reason is probably that we had to manually
        intervene, so make sure we tell the twiddler it was bad.
        """
        self.total_absolute_error = float('inf')
        self.crossing_state = STATE_START
        self.__finish_epoch()

    def step(self, target, error, sample_time):
        self.total_absolute_error += abs(error) * sample_time
        self.t += sample_time
        if target <= self.min_target:
            if self.crossing_state == STATE_OVER:
                self.crossing_state = STATE_UNDER
        elif target > self.min_target:
            if self.crossing_state == STATE_START:
                self.crossing_state = STATE_OVER
            elif self.crossing_state == STATE_UNDER:
                self.crossing_state = STATE_OVER
                self.__finish_epoch()

    def mean_absolute_error(self):
        if self.t == 0:
            return float('inf')
        return self.total_absolute_error / self.t

    def __start_epoch(self):
        self.t = 0
        self.total_absolute_error = 0

    def __finish_epoch(self):
        # Ignore short epochs; the current cycle is 25s.
        if self.t < 24:
            self.__start_epoch()
            return

        self.twiddler.twiddle(self.mean_absolute_error())
        self.pid.kp, self.pid.ki, self.pid.kd = self.twiddler.values
        rospy.loginfo("TuningPID: err=%.3f gain=%s delta=%s",
                      self.mean_absolute_error(),
                      self.twiddler.values,
                      self.twiddler.deltas)
        self.__start_epoch()
