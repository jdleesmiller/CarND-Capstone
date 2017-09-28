import rospy
import tf

from styx_msgs.msg import Lane, Waypoint


def make_lane(time, base_speed=10):
    lane = Lane()
    lane.header.frame_id = '/world'
    lane.header.stamp = rospy.Time(time)
    lane.waypoints = [
        make_waypoint(0.0, 0.0, speed=base_speed),
        make_waypoint(1.0, 0.0, speed=base_speed)
    ]
    return lane


def make_waypoint(x, y, z=0.0, yaw=0.0, speed=0.0):
    waypoint = Waypoint()
    waypoint.pose.pose.position.x = x
    waypoint.pose.pose.position.y = y
    waypoint.pose.pose.position.z = z
    waypoint.pose.pose.orientation = \
        tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    waypoint.twist.twist.linear.x = speed
    return waypoint
