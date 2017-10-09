#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from std_msgs.msg import Int32

import tf
import math

from waypoint_planner import WaypointPlanner
# PID Tuning: from stop_go_profiler import StopGoProfiler, make_stop_go_profile

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number


def kmph2mps(velocity_kmph):
    return (velocity_kmph * 1000.) / (60. * 60.)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        max_speed = kmph2mps(rospy.get_param('/waypoint_loader/velocity'))
        self.planner = WaypointPlanner(
            self.wait_for_base_waypoints(), max_speed)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.light_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

    def wait_for_base_waypoints(self):
        """
        The base waypoints never change (confirmed on Slack), so just listen for
        a single message to get our waypoints. This blocks until we get the
        message.

        TODO: Maybe we should just load the base waypoints in this node and get
        rid of the waypoint loader altogether, or change it into a ROS Service.
        """
        lane = rospy.wait_for_message('/base_waypoints', Lane)
        return lane.waypoints

    def pose_cb(self, msg):
        """
        Record the last known pose.
        """
        self.planner.position = msg.pose.position
        quat = (msg.pose.orientation.x, \
                msg.pose.orientation.y, \
                msg.pose.orientation.z, \
                msg.pose.orientation.w)
        self.planner.yaw = tf.transformations.euler_from_quaternion(quat)[2]

    def light_cb(self, msg):
        """
        Get traffic light state
        """
        self.planner.lights = msg.lights

    def publish_waypoints(self):
        # PID Tuning:
        # plan_waypoints = self.planner.find_closest_waypoints(LOOKAHEAD_WPS)
        plan_waypoints = self.planner.plan(LOOKAHEAD_WPS)
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

        start_time = 0
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
        # TODO: Callback for /traffic_waypoint message. Implement
        if msg.data != -1:
            self.planner.nextRedLight = msg.data
        else:
            self.planner.nextRedLight = len(self.planner.base_waypoints)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater().run()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
