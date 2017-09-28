import math
import sys
import unittest

sys.path.insert(0, '.')
from waypoint_utilities import make_lane

sys.path.insert(0, '..')
from stop_go_profiler import make_stop_go_profile, StopGoProfiler


def get_lane_speeds(lane):
    return [waypoint.twist.twist.linear.x for waypoint in lane.waypoints]


class TestStopGoProfiler(unittest.TestCase):

    def test_make_stop_go_profile(self):
        profile = make_stop_go_profile()

        # 2s hold, 10s accel, 5s hold, 5s decel
        duration = 2 + 10 + 5 + 5
        self.assertAlmostEqual(duration, profile.total_duration())

        # Using 2ad = vf^2 - vi^2:
        # 100/2 accel (a = 1), 5s*10m/s hold, 100/4 decel (a = -2)
        total_distance = 50 + 50 + 25
        self.assertAlmostEqual(total_distance, profile.total_distance())

        # Check speeds over time.
        self.assertAlmostEqual(0, profile.speed_at_time(0))
        self.assertAlmostEqual(0, profile.speed_at_time(1))
        self.assertAlmostEqual(0, profile.speed_at_time(2))
        self.assertAlmostEqual(1, profile.speed_at_time(3))
        self.assertAlmostEqual(2, profile.speed_at_time(4))
        self.assertAlmostEqual(10, profile.speed_at_time(12))
        self.assertAlmostEqual(10, profile.speed_at_time(17))
        self.assertAlmostEqual(8, profile.speed_at_time(18))
        self.assertAlmostEqual(2, profile.speed_at_time(21))
        self.assertAlmostEqual(0, profile.speed_at_time(22))

        # vf^2 = 2*a*d + vi^2, and a = 1
        self.assertAlmostEqual(0, profile.speed_at_distance(0))
        self.assertAlmostEqual(math.sqrt(2), profile.speed_at_distance(1))
        self.assertAlmostEqual(math.sqrt(98), profile.speed_at_distance(49))
        self.assertAlmostEqual(10, profile.speed_at_distance(50))
        self.assertAlmostEqual(10, profile.speed_at_distance(75))
        self.assertAlmostEqual(10, profile.speed_at_distance(100))

        # and now a = -2
        self.assertAlmostEqual(math.sqrt(96), profile.speed_at_distance(101))
        self.assertAlmostEqual(0, profile.speed_at_distance(125))

        # and it should wrap around
        self.assertAlmostEqual(math.sqrt(2), profile.speed_at_distance(126))

        # Check distance over time.
        self.assertAlmostEqual(0, profile.distance_at_time(0))
        self.assertAlmostEqual(0, profile.distance_at_time(1))
        self.assertAlmostEqual(0.5, profile.distance_at_time(3))
        self.assertAlmostEqual(2.0, profile.distance_at_time(4))
        self.assertAlmostEqual(total_distance,
                               profile.distance_at_time(duration))

        self.assertAlmostEqual(0, profile.initial_speed())
        self.assertAlmostEqual(0, profile.final_speed())

    def test_stop_go_profiler(self):
        start_time = 10
        profiler = StopGoProfiler(start_time, make_stop_go_profile())

        # If the time is before the start time, it wraps around.
        lane = make_lane(time=9)
        profiler.apply_speed_profile(lane)
        speeds = get_lane_speeds(lane)
        self.assertAlmostEqual(2.0, speeds[0])
        self.assertAlmostEqual(0.0, speeds[1])

        # Start of profile. It will set at 0 for 2s then start accelerating;
        # by the time it travels 1m, it is going sqrt(2) m/s.
        lane = make_lane(time=10)
        profiler.apply_speed_profile(lane)
        speeds = get_lane_speeds(lane)
        self.assertAlmostEqual(0.0, speeds[0])
        self.assertAlmostEqual(math.sqrt(2), speeds[1])

        # Middle of profile. It is accelerating; by the time it travels 1m, it
        # will have gained sqrt(2ad + vi^2)
        lane = make_lane(time=21)
        profiler.apply_speed_profile(lane)
        speeds = get_lane_speeds(lane)
        self.assertAlmostEqual(9.0, speeds[0])
        self.assertAlmostEqual(math.sqrt(2 + 81), speeds[1])

        # End of profile. It will stop and then wait until it starts
        # accelerating again, which is as above.
        lane = make_lane(time=32)
        profiler.apply_speed_profile(lane)
        speeds = get_lane_speeds(lane)
        self.assertAlmostEqual(0.0, speeds[0])
        self.assertAlmostEqual(math.sqrt(2), speeds[1])

        # If time is after profile end time, it wraps around. This time it is
        # starting with speed 1m/s, so the distance is sqrt(2ad + 1).
        lane = make_lane(time=35)
        profiler.apply_speed_profile(lane)
        speeds = get_lane_speeds(lane)
        self.assertAlmostEqual(1.0, speeds[0])
        self.assertAlmostEqual(math.sqrt(3), speeds[1])
