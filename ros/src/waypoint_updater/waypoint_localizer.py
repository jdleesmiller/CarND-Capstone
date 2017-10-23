import math
import rospy
import tf

from distance import find_position_distance


NEIGHBORHOOD_RADIUS = 5  # m


def find_yaw_from_orientation(orientation):
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    _roll, _pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    return yaw


def find_waypoint_heading(waypoint, position):
    return math.atan2(
        waypoint.pose.pose.position.y - position.y,
        waypoint.pose.pose.position.x - position.x)


class WaypointLocalizer(object):
    """
    Track the car's closest waypoint. Start searching from the car's last known
    waypoint, and only search in a small neighborhood of that position, for
    efficiency. If the vehicle's position is not in the neighborhood of its
    last known waypoint, then search the whole waypoint list.
    """
    def __init__(self, base_waypoints):
        self.base_waypoints = base_waypoints

        self.position = None
        self.yaw = None
        self.nearest_index = None

    def is_waiting(self):
        return self.position is None or self.nearest_index is None

    def update(self, pose):
        self.position = pose.position
        self.yaw = find_yaw_from_orientation(pose.orientation)

        # When initializing, start with a full scan.
        if self.nearest_index is None:
            self.__restart_localization()
            return

        self.nearest_index = self.__localize_in_neighborhood(
            NEIGHBORHOOD_RADIUS)

        if self.nearest_index is None:
            rospy.logwarn_throttle(
                1, 'waypoint_localizer: restarting localization')
            self.__restart_localization()

    def find_next_waypoint_index(self):
        """
        Find the index of the next waypoint ahead of the vehicle. This may be
        the nearest waypoint, or it may be the one after the nearest waypoint.
        If the nearest waypoint is the last one, and the vehicle is after the
        last waypoint, return None.
        """
        if self.is_waiting():
            return None

        heading = find_waypoint_heading(
            self.base_waypoints[self.nearest_index], self.position)

        if abs(heading - self.yaw) >= math.pi / 2.0:
            next_index = self.nearest_index + 1
        else:
            next_index = self.nearest_index

        if next_index >= len(self.base_waypoints):
            return None
        else:
            return next_index

    def __restart_localization(self):
        self.nearest_index = 0
        self.nearest_index = self.__localize_in_neighborhood(float('inf'))

    def __localize_in_neighborhood(self, max_distance):
        min_waypoint_index = None
        min_distance = float('inf')
        for i in range(self.nearest_index, len(self.base_waypoints)):
            distance_i = self.__find_distance_from_waypoint(i)
            if distance_i > max_distance:
                break
            if distance_i < min_distance:
                min_distance = distance_i
                min_waypoint_index = i
        return min_waypoint_index

    def __find_distance_from_waypoint(self, waypoint_index):
        return find_position_distance(
            self.position,
            self.base_waypoints[waypoint_index].pose.pose.position
        )
