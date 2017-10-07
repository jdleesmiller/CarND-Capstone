import rospy

from twiddler import Twiddler

DEFAULT_DELTA_FACTOR = 0.4

CONTROL_DELTA_PENALTY = 0.5
CONTROL_SQUARE_PENALTY = 0.1

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
        self.crossing_state = STATE_START
        self.__finish_epoch()

    def step(self, target, error, control, sample_time):
        self.total_absolute_error += abs(error) * sample_time
        self.total_control_cost += \
            self.__control_cost(control, sample_time) * sample_time
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

    def cost(self):
        if self.t == 0:
            return float('inf')
        cost = self.total_absolute_error + self.total_control_cost
        return cost / self.t

    def __control_cost(self, control, dt):
        if self.last_control is None:
            delta = 0
        else:
            delta = abs(control - self.last_control) / dt
        self.last_control = control
        return CONTROL_DELTA_PENALTY * delta + \
            CONTROL_SQUARE_PENALTY * control**2

    def __start_epoch(self):
        self.t = 0
        self.total_absolute_error = 0
        self.total_control_cost = 0
        self.last_control = None

    def __finish_epoch(self):
        # Ignore short epochs; the current cycle is 25s.
        if self.t < 24:
            self.__start_epoch()
            return

        rospy.logwarn("TuningPID: cost=%.3f (%.3f, %.3f) gain=%s delta=%s",
                      self.cost(),
                      self.total_absolute_error,
                      self.total_control_cost,
                      self.twiddler.values,
                      self.twiddler.deltas)
        self.twiddler.twiddle(self.cost())
        self.pid.kp, self.pid.ki, self.pid.kd = self.twiddler.values
        self.__start_epoch()
