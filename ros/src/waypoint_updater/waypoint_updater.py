#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane
from std_msgs.msg import Int32
from threading import Lock

from waypoint_localizer import WaypointLocalizer
from speed_estimator import SpeedEstimator
from waypoint_planner import WaypointPlanner
# PID Tuning: from stop_go_profiler import StopGoProfiler, make_stop_go_profile

'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.

As mentioned in the doc, you should ideally first implement a version which
does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status
of traffic lights too.
'''

# Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 30


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        base_waypoints = self.wait_for_base_waypoints()
        acceleration_limit = rospy.get_param('/dbw_node/accel_limit')
        deceleration_limit = rospy.get_param('/dbw_node/decel_limit')
        self.waypoint_localizer = WaypointLocalizer(base_waypoints)
        self.speed_estimator = SpeedEstimator()
        self.planner = WaypointPlanner(
            base_waypoints,
            self.waypoint_localizer,
            self.speed_estimator,
            acceleration_limit, deceleration_limit, LOOKAHEAD_WPS)
        self.planner_lock = Lock()

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

    def wait_for_base_waypoints(self):
        """
        The base waypoints never change (confirmed on Slack), so just listen
        for a single message to get our waypoints. This blocks until we get the
        message.
        """
        lane = rospy.wait_for_message('/base_waypoints', Lane)
        return lane.waypoints

    def pose_cb(self, msg):
        """
        Record the last known pose.
        """
        self.speed_estimator.update_pose(msg)
        with self.planner_lock:
            self.waypoint_localizer.update(msg.pose)

    def velocity_cb(self, msg):
        """
        Record the last known speed.
        """
        self.speed_estimator.update_speed(msg.twist.linear.x)

    def publish_waypoints(self):
        # PID Tuning:
        # plan_waypoints = self.planner.find_closest_waypoints(LOOKAHEAD_WPS)
        with self.planner_lock:
            plan_waypoints = self.planner.find_next_waypoints()
        if plan_waypoints is None:
            return
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.get_rostime()
        lane.waypoints = plan_waypoints
        # PID Tuning: self.profiler.apply_speed_profile(lane)
        self.final_waypoints_pub.publish(lane)

    def run(self):
        rate = rospy.Rate(10)

        start_time = 0.0
        while not start_time:
            start_time = rospy.get_time()

        # PID Tuning in the twist_controller: these will make the car stop
        # and go in a regular pattern to allow for PID gain tuning.
        # stop_go_profile = make_stop_go_profile(start_delay=5)
        # self.profiler = StopGoProfiler(start_time, stop_go_profile)

        while not rospy.is_shutdown():
            self.publish_waypoints()
            rate.sleep()

    def traffic_cb(self, msg):
        with self.planner_lock:
            self.planner.update_stop_index(msg.data)

if __name__ == '__main__':
    try:
        WaypointUpdater().run()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
