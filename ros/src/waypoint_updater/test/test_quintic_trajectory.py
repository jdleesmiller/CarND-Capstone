import numpy as np
import sys
import unittest

sys.path.insert(0, '..')
from quintic_trajectory import \
    QuinticTrajectory, min_jerk, \
    find_jmt_for_stop, find_jmt_for_change_in_speed


class TestQuinticTrajectory(unittest.TestCase):
    def test_constant_speed(self):
        # Speed is 2m/s
        t = QuinticTrajectory(np.array([0.0, 0.0, 0.0, 0.0, 2.0, 1.0]), 10.0)
        self.assertAlmostEqual(1.0, t.position(0.0))
        self.assertAlmostEqual(2.0, t.speed(0.0))

        self.assertAlmostEqual(3.0, t.position(1.0))
        self.assertAlmostEqual(2.0, t.speed(1.0))

        self.assertAlmostEqual(5.0, t.position(2.0))
        self.assertAlmostEqual(2.0, t.speed(2.0))

        self.assertIsNone(t.find_time_to_position(0.0))
        self.assertAlmostEqual(0.0, t.find_time_to_position(1.0))
        self.assertAlmostEqual(0.5, t.find_time_to_position(2.0))
        self.assertAlmostEqual(1.0, t.find_time_to_position(3.0))
        self.assertAlmostEqual(1.5, t.find_time_to_position(4.0))
        self.assertAlmostEqual(2.0, t.find_time_to_position(5.0))

        min_s, max_s = t.find_position_range()
        self.assertAlmostEqual(1.0, min_s)
        self.assertAlmostEqual(21.0, max_s)

        min_v, max_v = t.find_speed_range()
        self.assertAlmostEqual(2.0, min_v)
        self.assertAlmostEqual(2.0, max_v)

        min_a, max_a = t.find_acceleration_range()
        self.assertAlmostEqual(0.0, min_a)
        self.assertAlmostEqual(0.0, max_a)

    def test_constant_acceleration(self):
        # Accelerate at 2m/s^2
        t = QuinticTrajectory(np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 4.0)
        self.assertAlmostEqual(3.0, t.position(0.0))
        self.assertAlmostEqual(2.0, t.speed(0.0))

        self.assertAlmostEqual(6.0, t.position(1.0))
        self.assertAlmostEqual(4.0, t.speed(1.0))

        self.assertAlmostEqual(11.0, t.position(2.0))
        self.assertAlmostEqual(6.0, t.speed(2.0))

        self.assertIsNone(t.find_time_to_position(0.0))
        self.assertIsNone(t.find_time_to_position(2.0))
        self.assertAlmostEqual(0.0, t.find_time_to_position(3.0))
        self.assertAlmostEqual(1.0, t.find_time_to_position(6.0))
        self.assertAlmostEqual(2.0, t.find_time_to_position(11.0))

        min_s, max_s = t.find_position_range()
        self.assertAlmostEqual(3.0, min_s)
        self.assertAlmostEqual(27.0, max_s)

        min_v, max_v = t.find_speed_range()
        self.assertAlmostEqual(2.0, min_v)
        self.assertAlmostEqual(10.0, max_v)

        min_a, max_a = t.find_acceleration_range()
        self.assertAlmostEqual(2.0, min_a)
        self.assertAlmostEqual(2.0, max_a)

    def test_constant_jerk(self):
        # Start with acceleration -2m/s^2, and jerk at 6m/s^3 for 2s.
        t = QuinticTrajectory(np.array([0.0, 0.0, 1.0, -1.0, 3.0, 4.0]), 2.0)

        min_s, max_s = t.find_position_range()
        self.assertAlmostEqual(4.0, min_s)
        self.assertAlmostEqual(14.0, max_s)

        min_v, max_v = t.find_speed_range()
        self.assertAlmostEqual(8.0/3.0, min_v)
        self.assertAlmostEqual(11.0, max_v)

        min_a, max_a = t.find_acceleration_range()
        self.assertAlmostEqual(-2.0, min_a)
        self.assertAlmostEqual(10.0, max_a)

    def test_min_jerk(self):
        # Singular.
        t = min_jerk(0., 3., 2., 1., 6., 5., 4.)
        self.assertIsNone(t)

        # Feasible.
        t = min_jerk(10., 3., 2., 1., 6., 5., 4.)
        self.assertAlmostEqual(3.0, t.position(0.0))
        self.assertAlmostEqual(2.0, t.speed(0.0))
        self.assertAlmostEqual(6.0, t.position(10.0))
        self.assertAlmostEqual(5.0, t.speed(10.0))

        # Just check the ranges exist.
        t.find_position_range()
        t.find_speed_range()
        t.find_acceleration_range()

    def test_find_jmt_for_stop(self):
        # Speed up and then slow down.
        t = find_jmt_for_stop(0.0, 10.0, 10.0, 1.0, -1.0)

        # Checked graphically.
        self.assertAlmostEqual(7.6, t.horizon)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(-1.0, min_a)
        self.assertLessEqual(max_a, 1.0)

        # Again, but with more acceleration.
        t = find_jmt_for_stop(0.0, 10.0, 10.0, 2.0, -2.0)

        self.assertAlmostEqual(5.4, t.horizon)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(-2.0, min_a)
        self.assertLessEqual(max_a, 2.0)

        # Start from low speed and stop.
        t = find_jmt_for_stop(1.0, 10.0, 10.0, 2.0, -2.0)

        self.assertAlmostEqual(5.0, t.horizon)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(-2.0, min_a)
        self.assertLessEqual(max_a, 2.0)

        # Start from higher speed and stop.
        t = find_jmt_for_stop(5.0, 10.0, 10.0, 2.0, -2.0)

        self.assertAlmostEqual(3.9, t.horizon)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(-2.0, min_a)
        self.assertLessEqual(max_a, 2.0)

        # If we're going too fast, it's not possible to stop (without going
        # past it and then reversing).
        t = find_jmt_for_stop(10.0, 10.0, 10.0, 2.0, -2.0)
        self.assertIsNone(t)

    def test_find_jmt_for_change_in_speed(self):
        # Constant speed.
        t = find_jmt_for_change_in_speed(1.0, 1.0, 1.0, -1.0, margin=0.8)

        self.assertAlmostEqual(1.0, t.horizon)  # hard coded

        min_v, max_v = t.find_speed_range()
        self.assertAlmostEqual(1.0, min_v)
        self.assertAlmostEqual(1.0, max_v)

        # Speed up.
        t = find_jmt_for_change_in_speed(0.0, 10.0, 1.0, -1.0, margin=0.5)

        # Checked graphically.
        self.assertAlmostEqual(20, t.horizon)

        min_v, max_v = t.find_speed_range()
        self.assertAlmostEqual(0.0, min_v)
        self.assertAlmostEqual(10.0, max_v)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(0.0, min_a)
        self.assertLessEqual(max_a, 1.0)

        # Slow down.
        t = find_jmt_for_change_in_speed(10.0, 0.0, 1.0, -1.0, margin=0.5)

        # Checked graphically.
        self.assertAlmostEqual(20, t.horizon)

        min_a, max_a = t.find_acceleration_range()
        self.assertLessEqual(-1.0, min_a)
        self.assertLessEqual(max_a, 0.0)

if __name__ == '__main__':
    import rostest
    rostest.unitrun(
        'waypoint_updater',
        'test_quintic_trajectory', TestQuinticTrajectory)
