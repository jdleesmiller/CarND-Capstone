import math
import sys
import tf
import unittest

from geometry_msgs.msg import Pose, Quaternion

sys.path.insert(0, '.')
from waypoint_utilities import make_waypoint

sys.path.insert(0, '..')
from waypoint_localizer import WaypointLocalizer


def make_example_base_waypoints():
    return [
        make_waypoint(0.0, 0.0),
        make_waypoint(1.0, 0.0),
        make_waypoint(1.0, 1.0),
        make_waypoint(2.0, 2.0)
    ]


def make_pose(x, y, yaw):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.orientation = Quaternion(
        *tf.transformations.quaternion_from_euler(0., 0., yaw))
    return pose


class TestWaypointLocalizer(unittest.TestCase):

    def test_find_closest_waypoints_no_position(self):
        """
        If the vehicle's position hasn't been provided yet, return None.
        """
        localizer = WaypointLocalizer(make_example_base_waypoints())
        self.assertTrue(localizer.is_waiting())
        self.assertIsNone(localizer.find_next_waypoint_index())

    def test_find_closest_and_next_waypoints(self):
        """
        Return waypoints starting from the nearest point.
        """
        localizer = WaypointLocalizer(make_example_base_waypoints())

        localizer.update(make_pose(0.0, 0.0, 0.0))
        self.assertFalse(localizer.is_waiting())
        self.assertEqual(0, localizer.nearest_index)
        self.assertEqual(0, localizer.find_next_waypoint_index())

        localizer.update(make_pose(0.4, 0.0, 0))
        self.assertEqual(0, localizer.nearest_index)
        self.assertEqual(1, localizer.find_next_waypoint_index())

        localizer.update(make_pose(0.4, 0.0, math.pi))
        self.assertEqual(0, localizer.nearest_index)
        self.assertEqual(0, localizer.find_next_waypoint_index())

        localizer.update(make_pose(0.6, 0.0, 0.0))
        self.assertEqual(1, localizer.nearest_index)
        self.assertEqual(1, localizer.find_next_waypoint_index())

        localizer.update(make_pose(1.0, 0.1, math.pi / 2.0))
        self.assertEqual(1, localizer.nearest_index)
        self.assertEqual(2, localizer.find_next_waypoint_index())

        localizer.update(make_pose(1.9, 1.9, math.pi / 4))
        self.assertEqual(3, localizer.nearest_index)
        self.assertEqual(3, localizer.find_next_waypoint_index())

        localizer.update(make_pose(2.1, 2.1, math.pi / 4))
        self.assertEqual(3, localizer.nearest_index)
        self.assertIsNone(localizer.find_next_waypoint_index())


if __name__ == '__main__':
    import rostest
    rostest.unitrun(
        'waypoint_updater',
        'test_waypoint_localizer', TestWaypointLocalizer)
