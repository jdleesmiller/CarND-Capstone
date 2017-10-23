from distance import find_position_distance


class SpeedEstimator(object):
    """
    We should be able to get the current speed estimate directly from the
    `/current_velocity` topic. However, it's missing from some of the rosbags
    we need for testing, so estimate it from position if it is missing.

    Note: the estimator assumes the car is moving forward. Also, the
    estimates from position in the simulator are terrible due to apparent
    jitter in the message timestamps. Fortunately, it is not needed in the
    simulator.
    """
    def __init__(self, smooth=0.02):
        self.smooth = smooth

        self.measuring_directly = False
        self.speed = None
        self.position = None
        self.last_time = None

    def is_waiting(self):
        return self.speed is None

    def update_pose(self, msg):
        if self.measuring_directly:
            return

        position = msg.pose.position
        time = msg.header.stamp.to_sec()
        if self.position is None or self.last_time is None:
            self.position = position
            self.last_time = time
            return

        ds = find_position_distance(position, self.position)
        self.position = position

        dt = time - self.last_time
        self.last_time = time

        if dt <= 0:
            return

        speed = ds / dt
        if self.speed is None:
            self.speed = speed
        else:
            self.speed = self.smooth * speed + (1.0 - self.smooth) * self.speed

    def update_speed(self, speed):
        if not self.measuring_directly:
            self.measuring_directly = True
        self.speed = speed
