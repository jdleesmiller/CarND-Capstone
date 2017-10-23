import numpy as np


def min_jerk(T, si, siDot, siDDot, sf, sfDot, sfDDot):
    if T == 0:
        return None
    A = np.array([
        [1*T**3,  1*T**4,  1*T**5],
        [3*T**2,  4*T**3,  5*T**4],
        [6*T,    12*T**2, 20*T**3]])
    b = np.array([
        sf - (si + siDot*T + siDDot*T**2/2),
        sfDot - (siDot + siDDot*T),
        sfDDot - siDDot])
    try:
        x = np.linalg.solve(A, b)
        coefficients = np.hstack(([si, siDot, siDDot/2.0], x))
        return QuinticTrajectory(np.flipud(coefficients), T)
    except np.linalg.LinAlgError:
        return None


def find_distance(acceleration, initial_speed, final_speed):
    # 2ad = vf^2 - vi^2, so d = (vf^2 - vi^2) / 2a
    if acceleration == 0.0:
        return (final_speed - initial_speed) * float('inf')
    return (final_speed**2 - initial_speed**2) / 2.0 / acceleration


def find_jmt_for_change_in_speed(
        initial_speed, final_speed, acceleration_limit, deceleration_limit,
        margin, tolerance=0.1):
    """
    Use the basic infinite-jerk kinematics equations to find the time and
    distance it should take to effect the given change of speed, and then
    fit a jerk minimizing trajectory using those parameters.

    The 'margin' reduces the average acceleration for the infinite jerk
    case to give the JMT some slack to work with. If the margin is 1.0,
    the peak acceleration for the JMT will exceed the acceleration or
    deceleration limits.
    """

    # If we're already at or very close to the target speed, just
    # return the final speed.
    delta_speed = final_speed - initial_speed
    if abs(delta_speed) < tolerance:
        return QuinticTrajectory(
            np.array([0.0, 0.0, 0.0, 0.0, final_speed, 0.0]), 1.0)

    if delta_speed > 0.0:
        acceleration = acceleration_limit
    else:
        acceleration = deceleration_limit
    acceleration *= margin
    distance = find_distance(acceleration, initial_speed, final_speed)
    time = delta_speed / acceleration

    return min_jerk(
        time,
        0.0, initial_speed, 0.0,
        distance, final_speed, 0.0)


def find_jmt_for_stop(
        initial_speed, distance,
        speed_limit, acceleration_limit, deceleration_limit,
        delta_time=0.1, tolerance=0.1, max_time=60.0):
    """
    Find a jerk minimizing trajectory that will stop the vehicle after the
    given distance. The unknown is how long it should take, so just try
    giving the vehicle longer and longer time until the acceleration and
    deceleration constraints are satisfied.

    It's also necessary to constrain speed to avoid situations where the
    vehicle can stop sooner by speeding up and then slowing down; the speed
    limit should typically be set just above the vehicle's current speed.
    """

    time = delta_time
    while True:
        jmt = min_jerk(
            time,
            0.0, initial_speed, 0.0,
            distance, 0.0, 0.0)

        if time > max_time:
            return None

        time += delta_time

        if jmt is None:
            continue

        # If we are going too fast, allow more time; if we're going backwards,
        # more time won't help, so give up.
        min_speed, max_speed = jmt.find_speed_range()
        if min_speed < -tolerance:
            return None
        if max_speed > speed_limit + tolerance:
            continue

        # If we aren't hitting the acceleration target, allow more time.
        min_acceleration, max_acceleration = jmt.find_acceleration_range()
        if min_acceleration < deceleration_limit:
            continue
        if max_acceleration > acceleration_limit:
            continue

        return jmt


class QuinticTrajectory(object):
    """
    Find and work with a trajectory represented as a qunitic polynomial,
    typically a jerk-minimizing one.

    Coefficients are in the order for np.polyval (highest power first).
    """
    def __init__(self, coefficients, horizon):
        self.position = np.poly1d(coefficients, variable='x')
        self.speed = self.position.deriv()
        self.acceleration = self.speed.deriv()
        self.jerk = self.acceleration.deriv()
        self.horizon = horizon

    def __str__(self):
        return str(self.position)

    def find_time_to_position(self, position):
        roots = (self.position - position).roots
        feasible_roots = self.__find_feasible_roots(roots)
        if feasible_roots.size == 0:
            return None
        else:
            return np.min(feasible_roots)

    def find_position_range(self):
        return self.__find_roots_range(self.speed.roots, self.position)

    def find_speed_range(self):
        return self.__find_roots_range(self.acceleration.roots, self.speed)

    def find_acceleration_range(self):
        return self.__find_roots_range(self.jerk.roots, self.acceleration)

    def __find_roots_range(self, roots, value):
        feasible_roots = list(self.__find_feasible_roots(roots))
        values = [value(t) for t in feasible_roots + [0.0, self.horizon]]
        return min(values), max(values)

    def __find_feasible_roots(self, roots):
        real_roots = np.real(roots[np.isreal(roots)])
        return real_roots[(real_roots >= 0.0) & (real_roots <= self.horizon)]
