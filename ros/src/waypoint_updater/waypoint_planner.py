import rospy

from distance import find_path_distance
from plan import Plan, get_waypoint_speed
from quintic_trajectory import find_jmt_for_change_in_speed, find_jmt_for_stop

# See notes for find_jmt_for_change_in_speed.
MARGIN = 1.0

# Leave some time for planning latency when calculating how much distance the
# car has to stop.
LATENCY = 0.1  # s

# Stop this much before the stop line; the car uses the center of the rear
# axle as its origin, and we don't want the front of the car sticking out
# into the intersection.
VEHICLE_LENGTH = 4.0  # m


class WaypointPlanner(object):
    """
    Update the car's plan in response to changes in the trafic light stop
    index and provide the car with the next set of waypoints it should
    traverse.
    """
    def __init__(self, base_waypoints,
                 localizer, speed_estimator,
                 acceleration_limit, deceleration_limit,
                 num_waypoints):
        self.base_waypoints = base_waypoints
        self.localizer = localizer
        self.speed_estimator = speed_estimator
        self.acceleration_limit = acceleration_limit
        self.deceleration_limit = deceleration_limit
        self.num_waypoints = num_waypoints

        self.stop_index = None
        self.plan = None

    def is_waiting(self):
        """
        Planner is still waiting for position and speed data.
        """
        return self.localizer.is_waiting() or \
            self.speed_estimator.is_waiting()

    def update_stop_index(self, stop_index):
        """
        Called with the index of the waypoint to stop at when there is a
        traffic light, or -1 if there is no traffic light.
        """
        if self.is_waiting():
            return

        # If we've already planned for this waypoint, stick to the plan.
        if self.stop_index == stop_index:
            return
        self.stop_index = stop_index

        self.__plan()

    def find_next_waypoints(self):
        """
        Find the next list of waypoints for the car to traverse.
        """
        if self.is_waiting() or self.plan is None:
            return None

        next_waypoint_index = self.localizer.find_next_waypoint_index()

        # If we're past the last waypoint, there won't be a next waypoint.
        if next_waypoint_index is None:
            return None

        # If we've gone backwards or wrapped around, we need a new plan.
        if next_waypoint_index < self.plan.start_index - 1:
            rospy.logwarn('waypoint_updater: replanning from wp %d (start=%d)',
                          next_waypoint_index, self.plan.start_index)
            self.__plan()

        self.plan.advance(next_waypoint_index)
        return self.plan.waypoints

    def __plan(self):
        next_waypoint_index = self.localizer.find_next_waypoint_index()

        if next_waypoint_index is None:
            # If we're past the last waypoint, make a plan that just includes
            # the last waypoint, which should have zero speed.
            rospy.loginfo('waypoint_updater: plan stop after base waypoints')
            self.plan = self.__make_plan(self.localizer.nearest_index, None)
        elif self.stop_index < 0:
            rospy.loginfo(
                'waypoint_updater: plan free flow from v=%.1fm/s',
                self.speed_estimator.speed)
            self.plan = self.__plan_freeflow(next_waypoint_index)
        else:
            rospy.loginfo(
                'waypoint_updater: plan stop from v=%.1fm/s at waypoint %d',
                self.speed_estimator.speed, self.stop_index)
            self.plan = self.__plan_stop(next_waypoint_index, self.stop_index)

    def __plan_freeflow(self, start_index):
        initial_speed = self.speed_estimator.speed
        final_speed = self.__find_speed_limit(start_index)

        jmt = find_jmt_for_change_in_speed(
            initial_speed, final_speed,
            self.acceleration_limit, self.deceleration_limit, MARGIN)

        if jmt is None:
            rospy.logwarn('waypoint_updater: free flow plan failed')

        return self.__make_plan(start_index, jmt)

    def __plan_stop(self, start_index, stop_index):
        plan_stop_index = stop_index - start_index

        # If the stop index is behind us, just keep going.
        if plan_stop_index < 0:
            rospy.logwarn('waypoint_updater: already past stop line')
            return self.__plan_freeflow(start_index)

        initial_speed = self.speed_estimator.speed
        speed_limit = max(
            initial_speed,
            self.__find_speed_limit(start_index))

        distance = find_path_distance(
            self.base_waypoints, start_index, stop_index)
        latency_distance = initial_speed * LATENCY
        distance -= latency_distance + VEHICLE_LENGTH

        jmt = find_jmt_for_stop(
            initial_speed, distance,
            speed_limit, self.acceleration_limit, self.deceleration_limit)

        if jmt is None:
            # If it's too late to stop, just keep going.
            rospy.logwarn(
                'waypoint_updater: too late to stop (%.1fm)',
                distance)
            return self.__plan_freeflow(start_index)
        else:
            rospy.loginfo(
                'waypoint_updater: stopping in %.1fm over %.1fs',
                distance, jmt.horizon)
            return self.__make_plan(start_index, jmt)

    def __find_speed_limit(self, start_index):
        return max(
            get_waypoint_speed(waypoint)
            for waypoint in self.base_waypoints[start_index:]
        )

    def __make_plan(self, start_index, trajectory):
        return Plan(
            self.base_waypoints,
            start_index,
            self.num_waypoints,
            trajectory)
