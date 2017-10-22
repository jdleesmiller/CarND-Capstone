from math import atan


class YawController(object):

    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

    def get_angle(self, radius):
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        # linear_velocity: target linear velocity at end of timestep (m/s)
        # angular_velocity: target angular velocity (rad/s)
        # current_velocity: current linear velocity (m/s)

        # If we're accelerating, we don't need to turn as quickly to have the
        # same change in angle.
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.

        # If we're going fast, we can't turn as quickly without going over
        # the lateral acceleration limit.
        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity)
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

        # Over a given timestep, the linear speed gives us an arc length and
        # the angular velocity a subtended angle, so l = r * theta, and
        # r = l / theta is the radius of curvature.
        return self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0
