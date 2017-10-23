import math


def find_squared_position_distance(position0, position1):
    """
    Find squared distance between two positions.
    """
    delta_x = position1.x - position0.x
    delta_y = position1.y - position0.y
    delta_z = position1.z - position0.z
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z


def find_position_distance(position0, position1):
    """
    Find distance between two positions.
    """
    return math.sqrt(find_squared_position_distance(position0, position1))


def find_waypoint_distance(waypoint0, waypoint1):
    return find_position_distance(
        waypoint0.pose.pose.position,
        waypoint1.pose.pose.position)


def find_path_distance(waypoints, start, stop):
    path_distance = 0
    for i in range(start, stop - 1):
        path_distance += find_waypoint_distance(waypoints[i], waypoints[i + 1])
    return path_distance
