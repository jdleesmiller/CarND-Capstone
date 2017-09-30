import sys
import unittest

from geometry_msgs.msg import Vector3

sys.path.insert(0, '.')
from waypoint_utilities import make_waypoint

sys.path.insert(0, '..')
from waypoint_planner import WaypointPlanner


def make_example_base_waypoints():
    return [
        make_waypoint(0.0, 0.0),
        make_waypoint(1.0, 0.0),
        make_waypoint(1.0, 1.0),
        make_waypoint(2.0, 2.0)
    ]


class TestWaypointPlanner(unittest.TestCase):

    def test_find_closest_waypoints_no_position(self):
        """
        If the vehicle's position hasn't been provided yet, return None.
        """
        planner = WaypointPlanner(make_example_base_waypoints())
        self.assertIsNone(planner.find_closest_waypoints(1))

    def test_find_closest_waypoints_nearest(self):
        """
        Return waypoints starting from the nearest point.
        """
        planner = WaypointPlanner(make_example_base_waypoints())

        planner.position = Vector3(0, 0, 0)
        waypoints = planner.find_closest_waypoints(1)
        self.assertEqual(1, len(waypoints))
        self.assertEqual(0, waypoints[0].pose.pose.position.x)
        self.assertEqual(0, waypoints[0].pose.pose.position.y)
        self.assertEqual(0, waypoints[0].pose.pose.position.z)

        planner.position = Vector3(0.9, 0.9, 0)
        waypoints = planner.find_closest_waypoints(2)
        self.assertEqual(2, len(waypoints))
        self.assertEqual(1, waypoints[0].pose.pose.position.x)
        self.assertEqual(1, waypoints[0].pose.pose.position.y)
        self.assertEqual(2, waypoints[1].pose.pose.position.x)
        self.assertEqual(2, waypoints[1].pose.pose.position.y)

        # Check it wraps back around to the start.
        planner.position = Vector3(0.9, 0.9, 0)
        waypoints = planner.find_closest_waypoints(3)
        self.assertEqual(3, len(waypoints))
        self.assertEqual(1, waypoints[0].pose.pose.position.x)
        self.assertEqual(1, waypoints[0].pose.pose.position.y)
        self.assertEqual(2, waypoints[1].pose.pose.position.x)
        self.assertEqual(2, waypoints[1].pose.pose.position.y)
        self.assertEqual(0, waypoints[2].pose.pose.position.x)
        self.assertEqual(0, waypoints[2].pose.pose.position.y)


if __name__ == '__main__':
    import rostest
    rostest.unitrun(
        'waypoint_updater',
        'test_waypoint_planner', TestWaypointPlanner)
