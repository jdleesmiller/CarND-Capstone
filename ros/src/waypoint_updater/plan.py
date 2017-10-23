from collections import deque
from copy import deepcopy

from distance import find_waypoint_distance


def get_waypoint_speed(waypoint):
    return waypoint.twist.twist.linear.x


def set_waypoint_speed(waypoint, speed):
    waypoint.twist.twist.linear.x = speed


class Plan(object):
    def __init__(self, base_waypoints, start_index, num_waypoints, trajectory):
        self.base_waypoints = base_waypoints
        self.start_index = start_index
        self.num_waypoints = num_waypoints
        self.trajectory = trajectory

        self.final_speed = float('inf')
        self.waypoints = deque()
        self.end_index = start_index
        self.end_distance = 0.0
        self.__fill()
        self.__spur_on_if_needed()

    def advance(self, waypoint_index):
        delta = waypoint_index - self.start_index
        self.start_index += delta

        while delta > 0 and len(self.waypoints) > 0:
            delta -= 1
            self.waypoints.popleft()
            self.__fill()

    def get_speeds(self):
        return [get_waypoint_speed(waypoint) for waypoint in self.waypoints]

    def __fill(self):
        while not self.__done() and not self.__full():
            waypoint = deepcopy(self.base_waypoints[self.end_index])
            self.end_index += 1

            if len(self.waypoints) > 0:
                self.end_distance += find_waypoint_distance(
                    self.waypoints[-1], waypoint)

            if self.trajectory is not None:
                time = self.trajectory.find_time_to_position(self.end_distance)
                if time is None:
                    self.final_speed = self.trajectory.speed(
                        self.trajectory.horizon)
                    self.trajectory = None
                else:
                    speed = self.trajectory.speed(time)

            if self.trajectory is None:
                speed = self.final_speed

            if speed < get_waypoint_speed(waypoint):
                set_waypoint_speed(waypoint, speed)
            self.waypoints.append(waypoint)

    def __done(self):
        return self.end_index >= len(self.base_waypoints)

    def __full(self):
        return len(self.waypoints) >= self.num_waypoints

    def __spur_on_if_needed(self):
        """
        The car often needs some encouragement to get started.
        """
        if len(self.waypoints) < 2:
            return
        next_speed = (get_waypoint_speed(self.waypoints[0]) +
                      get_waypoint_speed(self.waypoints[1])) / 2.0
        set_waypoint_speed(self.waypoints[0], next_speed)
