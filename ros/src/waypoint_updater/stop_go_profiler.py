import math


def find_position_distance(position0, position1):
    """
    Distance between two positions.
    """
    delta_x = position1.x - position0.x
    delta_y = position1.y - position0.y
    delta_z = position1.z - position0.z
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def find_waypoint_distance(waypoint0, waypoint1):
    return find_position_distance(
        waypoint0.pose.pose.position,
        waypoint1.pose.pose.position)


class InfiniteJerkSpeedProfileSegment(object):
    """
    A segment of a speed profile with constant acceleration.
    """
    def __init__(self, initial_speed, acceleration, duration):
        final_speed = initial_speed + acceleration * duration

        self.initial_speed = initial_speed
        self.final_speed = final_speed
        self.acceleration = acceleration
        self.duration = duration
        self.distance = self.distance_at_time(duration)

    def speed_at_time(self, time):
        return self.initial_speed + self.acceleration * time

    def distance_at_time(self, time):
        if self.acceleration == 0:
            return self.initial_speed * time

        # 2ad = vf^2 - vi^2
        speed = self.speed_at_time(time)
        return (speed ** 2 - self.initial_speed ** 2) / 2.0 / self.acceleration

    def speed_at_distance(self, distance):
        # vf^2 = 2ad + vi^2
        return math.sqrt(
            2 * self.acceleration * distance + self.initial_speed ** 2)


class InfiniteJerkSpeedProfile(object):
    """
    Speed profile assuming that acceleration is piecewise-constant.
    """
    def __init__(self, segments=None):
        if segments is None:
            segments = []
        self.segments = segments

    def add_segment(self, acceleration, duration):
        self.segments.append(
            InfiniteJerkSpeedProfileSegment(
                self.final_speed(), acceleration, duration))

    def total_duration(self):
        return sum(segment.duration for segment in self.segments)

    def total_distance(self):
        return sum(segment.distance for segment in self.segments)

    def initial_speed(self):
        if len(self.segments) == 0:
            return 0
        else:
            return self.segments[0].initial_speed

    def final_speed(self):
        if len(self.segments) == 0:
            return 0
        else:
            return self.segments[-1].final_speed

    def speed_at_time(self, time):
        if time >= 0:
            for segment in self.segments:
                if time <= segment.duration:
                    return segment.speed_at_time(time)
                time -= segment.duration
        return float('nan')

    def distance_at_time(self, time):
        if time >= 0:
            distance = 0
            for segment in self.segments:
                if time <= segment.duration:
                    return distance + segment.distance_at_time(time)
                distance += segment.distance
                time -= segment.duration

        return float('nan')

    def speed_at_distance(self, target_distance):
        if target_distance <= 0:
            return self.initial_speed()

        target_distance = target_distance % self.total_distance()

        for segment in self.segments:
            if target_distance < segment.distance:
                return segment.speed_at_distance(target_distance)
            target_distance -= segment.distance

        return float('nan')


def make_stop_go_profile(
        start_delay=2,    # s to remain parked
        acceleration=1,   # m/s^2
        target_speed=10,  # m/s
        hold_delay=5,     # s to remain at target speed
        deceleration=2):   # m/s^2
    """
    Make a simple trapezoidal speed profile that goes from zero to the target
    speed and back to zero again.
    """

    profile = InfiniteJerkSpeedProfile()
    profile.add_segment(0, start_delay)
    profile.add_segment(acceleration, target_speed / acceleration)
    profile.add_segment(0, hold_delay)
    profile.add_segment(-deceleration, target_speed / deceleration)

    return profile


def set_waypoint_speed(waypoint, speed):
    waypoint.twist.twist.linear.x = speed


class StopGoProfiler(object):

    def __init__(self, start_time, speed_profile):
        self.start_time = start_time
        self.speed_profile = speed_profile

    def apply_speed_profile(self, lane):
        if len(lane.waypoints) == 0:
            return

        profile_time = (lane.header.stamp.to_sec() - self.start_time) % \
            self.speed_profile.total_duration()
        set_waypoint_speed(
            lane.waypoints[0],
            self.speed_profile.speed_at_time(profile_time))

        distance = self.speed_profile.distance_at_time(profile_time)
        for waypoint_index in range(len(lane.waypoints) - 1):
            distance += find_waypoint_distance(
                lane.waypoints[waypoint_index],
                lane.waypoints[waypoint_index + 1])

            set_waypoint_speed(
                lane.waypoints[waypoint_index + 1],
                self.speed_profile.speed_at_distance(distance))
