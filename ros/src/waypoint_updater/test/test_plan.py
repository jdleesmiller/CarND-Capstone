import numpy as np
import sys
import unittest

sys.path.insert(0, '.')
from waypoint_utilities import make_waypoint

sys.path.insert(0, '..')
from plan import Plan
from quintic_trajectory import QuinticTrajectory


def make_example_base_waypoints(n, speed):
    return [
        make_waypoint(float(x), float(x), speed=speed)
        for x in range(0, n)
    ]


class TestPlan(unittest.TestCase):
    def test_plan_to_speed_limit(self):
        # Accelerate at 1m/s^2 for 10s
        base_waypoints = make_example_base_waypoints(n=100, speed=5.0)
        trajectory = QuinticTrajectory(
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), 10.0)
        plan = Plan(
            base_waypoints,
            start_index=0,
            num_waypoints=3,
            trajectory=trajectory)

        self.assertEqual(3, len(plan.waypoints))
        self.assertEqual(0, plan.start_index)
        self.assertEqual(3, plan.end_index)

        self.assertEqual(base_waypoints[0].pose, plan.waypoints[0].pose)
        self.assertEqual(base_waypoints[1].pose, plan.waypoints[1].pose)
        self.assertEqual(base_waypoints[2].pose, plan.waypoints[2].pose)

        # Should have made a copy of the waypoints to overwrite twist.
        self.assertNotEqual(base_waypoints[0].twist, plan.waypoints[0].twist)

        # Should be speeding up and not starting from zero speed.
        speeds_0 = plan.get_speeds()
        self.assertLess(0.5, speeds_0[0])
        self.assertLessEqual(speeds_0[0], speeds_0[1])
        self.assertLessEqual(speeds_0[1], speeds_0[2])

        # Should not advance if waypoint matches start index.
        plan.advance(0)
        self.assertEqual(0, plan.start_index)
        self.assertEqual(3, plan.end_index)
        self.assertEqual(base_waypoints[0].pose, plan.waypoints[0].pose)

        # Advance one waypoint. Should pick the next one from base.
        plan.advance(1)
        self.assertEqual(1, plan.start_index)
        self.assertEqual(4, plan.end_index)
        self.assertEqual(base_waypoints[1].pose, plan.waypoints[0].pose)
        speeds_1 = plan.get_speeds()
        self.assertEqual(speeds_0[1], speeds_1[0])
        self.assertEqual(speeds_0[2], speeds_1[1])
        self.assertLess(speeds_1[1], speeds_1[2])

        # Try another no-op advance.
        plan.advance(1)
        self.assertEqual(1, plan.start_index)
        self.assertEqual(4, plan.end_index)

        # Advance two waypoints at once.
        plan.advance(3)
        self.assertEqual(3, plan.start_index)
        self.assertEqual(6, plan.end_index)
        self.assertEqual(base_waypoints[3].pose, plan.waypoints[0].pose)

        # Advance several waypoints at once. This is enough to hit the speed
        # limit.
        plan.advance(10)
        self.assertEqual(10, plan.start_index)
        self.assertEqual(13, plan.end_index)
        self.assertEqual(base_waypoints[10].pose, plan.waypoints[0].pose)

        self.assertEqual(5.0, min(plan.get_speeds()))
        self.assertEqual(5.0, max(plan.get_speeds()))

    def test_plan_to_trajectory_end(self):
        # Accelerate at 1m/s^2 for 10s
        base_waypoints = make_example_base_waypoints(n=100, speed=50.0)
        trajectory = QuinticTrajectory(
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), 10.0)
        plan = Plan(
            base_waypoints,
            start_index=0,
            num_waypoints=5,
            trajectory=trajectory)

        self.assertEqual(5, len(plan.waypoints))
        self.assertEqual(0, plan.start_index)
        self.assertEqual(5, plan.end_index)

        # It should take 50m to get up to 10m/s at 1m/s^2, and each waypoint
        # get us sqrt(2) ~= 1.41, so it takes just over 35 waypoints to get up
        # to speed.
        plan.advance(35)
        self.assertGreater(10.0, min(plan.get_speeds()))
        self.assertEqual(10.0, max(plan.get_speeds()))
        self.assertAlmostEqual(10.0, plan.final_speed)

        # It should now be using the final speed from the trajectory.
        self.assertIsNone(plan.trajectory)

        plan.advance(36)
        self.assertEqual(10.0, min(plan.get_speeds()))
        self.assertEqual(10.0, max(plan.get_speeds()))

    def test_plan_to_base_waypoints_end(self):
        # Accelerate at 1m/s^2 for 10s
        base_waypoints = make_example_base_waypoints(n=100, speed=50.0)
        trajectory = QuinticTrajectory(
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), 10.0)
        plan = Plan(
            base_waypoints,
            start_index=70,
            num_waypoints=5,
            trajectory=trajectory)

        self.assertEqual(5, len(plan.waypoints))
        self.assertEqual(70, plan.start_index)
        self.assertEqual(75, plan.end_index)

        plan.advance(95)
        self.assertEqual(5, len(plan.waypoints))
        self.assertEqual(95, plan.start_index)
        self.assertEqual(100, plan.end_index)
        self.assertEqual(base_waypoints[95].pose, plan.waypoints[0].pose)
        self.assertEqual(base_waypoints[99].pose, plan.waypoints[4].pose)

        plan.advance(96)
        self.assertEqual(4, len(plan.waypoints))
        self.assertEqual(96, plan.start_index)
        self.assertEqual(100, plan.end_index)
        self.assertEqual(base_waypoints[96].pose, plan.waypoints[0].pose)
        self.assertEqual(base_waypoints[99].pose, plan.waypoints[3].pose)

        plan.advance(98)
        self.assertEqual(2, len(plan.waypoints))
        self.assertEqual(98, plan.start_index)
        self.assertEqual(100, plan.end_index)
        self.assertEqual(base_waypoints[98].pose, plan.waypoints[0].pose)
        self.assertEqual(base_waypoints[99].pose, plan.waypoints[1].pose)

        plan.advance(99)
        self.assertEqual(1, len(plan.waypoints))
        self.assertEqual(99, plan.start_index)
        self.assertEqual(100, plan.end_index)
        self.assertEqual(base_waypoints[99].pose, plan.waypoints[0].pose)

        plan.advance(100)
        self.assertEqual(0, len(plan.waypoints))
        self.assertEqual(100, plan.start_index)
        self.assertEqual(100, plan.end_index)

if __name__ == '__main__':
    import rostest
    rostest.unitrun(
        'waypoint_updater',
        'test_plan', TestPlan)
