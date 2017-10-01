#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_line_waypoints = []
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.last_known_wp = -1

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # go through all traffic lights stop lines
        for stop_line in stop_line_positions:
            stop_line_pose = PoseStamped()
            stop_line_pose.pose.position.x = stop_line[0]
            stop_line_pose.pose.position.y = stop_line[1]
            stop_line_pose.pose.position.z = 0
            stop_line_pose.pose.orientation = 0

            # get nearest wp to light
            stop_line_wp = self.get_closest_waypoint(stop_line_pose)
            self.stop_line_waypoints.append(stop_line_wp)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_squared_distance(self, a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = 1e8
        min_wp = -1

        if self.waypoints is not None:

            # store last waypoint as starting point for search
            i = self.last_known_wp
            while True: 

                # determine distance
                dist = self.get_squared_distance(self.waypoints.waypoints[i].pose.pose.position,
                                                 pose.pose.position)

                # if waypoint is closer than previous, continue. Otherwise this should be the closest since
                # waypoints are in order of path
                if dist < min_dist:
                    min_dist = dist
                    min_wp = i
                else:
                    break

                # if last waypoint, start from beginning of list
                if i == (len(self.waypoints.waypoints)-1):
                    i = 0
                else:
                    i += 1

        return min_wp
    
    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light_wp):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get detection and classification
        state = self.light_classifier.get_classification(cv_image)

        if state == TrafficLight.UNKNOWN:
            # Use ground truth if available
            nearest = 1e8
            light_wp_position = self.waypoints.waypoints[light_wp].pose.pose.position
            for light in self.lights:
                # get nearest light to car position
                dist = self.get_squared_distance(light_wp_position, light.pose.pose.position)

                if dist < nearest:
                    # update nearest waypoint distance
                    nearest = dist
                    state = light.state

        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_wp = -1
        state = TrafficLight.UNKNOWN

        if self.pose and self.waypoints:

            
            vehicle_wp = self.get_closest_waypoint(self.pose)
            self.last_known_wp = vehicle_wp
            nearest_dist = 1e8

            # go through all traffic lights stop lines
            for stop_line_wp in self.stop_line_waypoints:

                if stop_line_wp < vehicle_wp:
                    wp_dist = stop_line_wp + (len(self.waypoints.waypoints) - vehicle_wp)
                else:
                    wp_dist = stop_line_wp - vehicle_wp

                # if wp index distance is less than current nearest, set as nearest
                if wp_dist < nearest_dist:
                    # update nearest waypoint distance
                    nearest_dist = wp_dist
                    # set to nearest index
                    light_wp = stop_line_wp

            if light_wp != -1:
                state = self.get_light_state(light_wp)
                #rospy.loginfo("Traffic Light Ahead: wp=%d state=%d dist=%d", light_wp, state, nearest_dist)

        return light_wp, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
