import math
import tf
import numpy as np
import rospy
import yaml
import copy

targetSpeed = 10 # m/s ---- 4.4074 m/s = 10 MPH
accelDist = 10 # Distance over which acceleration should occur
decelDist = 100 # Distance over which deceleration should occur

# Number of waypoints before the traffic light waypoint to stop
STOP_EARLY_DIST = 28

def minJerk(T, si, siDot, siDDot, sf, sfDot, sfDDot):
    if T == 0:
        #print("Passed zero time")
        return None
    A = np.array([[  T**3,    T**4,    T**5], \
                  [3*T**2,  4*T**3,  5*T**4], \
                  [6*T   , 12*T**2, 20*T**3]])
    b = np.array([sf     - (si    +  siDot*T + siDDot*T**2/2), \
                  sfDot  - (siDot + siDDot*T                ), \
                  sfDDot -  siDDot                          ])
    if np.linalg.cond(A) < 100000: # Check that matrix is not singular
        x = np.linalg.solve(A,b)
        return np.hstack(([si,siDot,siDDot/2],x))
    else:
        #print("Yikes", np.linalg.cond(A),T, si, siDot, siDDot, sf, sfDot, sfDDot)
        return None

def position_distance(position0, position1):
    """
    Distance between two positions.
    """
    delta_x = position1.x - position0.x
    delta_y = position1.y - position0.y
    delta_z = position1.z - position0.z
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

def getClosestForwardLight(lights, position, yaw):
    # Return index of closest traffic light given list of all lights, and current position/yaw
    # This will be unnecesary once /traffic_waypoint returns the waypoint index of the next red light
    min_distance = float('inf')
    min_index = 0
    for index in range(len(lights)):
        lightPos = lights[index].pose.pose.position
        distance = position_distance(position, lightPos)
        if distance < min_distance:
            min_distance = distance
            min_index = index
    head = np.arctan2(lights[min_index].pose.pose.position.y - position.y , \
                      lights[min_index].pose.pose.position.x - position.x)
    if np.abs(head - yaw) > np.pi/2 or min_distance < 22:
        min_index += 1
        if min_index >= len(lights):
            min_index -= len(lights)
    return min_index

class WaypointPlanner(object):

    def __init__(self, base_waypoints):
        self.base_waypoints = base_waypoints
        for i in range(len(base_waypoints)):
            self.set_waypoint_velocity(self.base_waypoints,i,0)
        self.position = None
        self.yaw = None
        self.lights = None
        self.targetWaypoint = None
        self.initialize = True
        self.lastWaypoint = 0

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        if wp1 == wp2:
            return 0
        if wp2 >= len(waypoints):
            wp2 -= len(waypoints)
        if wp2 < 0:
            wp2 += len(waypoints)
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        wp = copy.copy(wp1)
        go = True
        while go:
            if wp+1 >= len(waypoints):
                dist += dl(waypoints[wp].pose.pose.position, waypoints[0].pose.pose.position)
                wp = 0
            else:
                dist += dl(waypoints[wp].pose.pose.position, waypoints[wp+1].pose.pose.position)
                wp += 1
            go = wp != wp2
        return dist

    def getJMT(self, currentSpeed, WPs, min_index, light_index, distToLight, num_waypoints, accelerate):
        # Uses jerk minimizing trajectory to determine speeds for wayPoints WPs
        if min_index+1 == len(WPs):
            vNext = self.get_waypoint_velocity(WPs[0]) # m/s
        else:
            vNext = self.get_waypoint_velocity(WPs[min_index+1]) # m/s
        dv = vNext - currentSpeed # m/s
        dx = self.distance(WPs,min_index,min_index+1) # m
        if np.abs(vNext+currentSpeed) < 0.01: # Are at a stop
            a = 0
            T = distToLight/(targetSpeed/2)
        else:
            dt = dx/((vNext+currentSpeed)/2) # s (dx = vdt)
            avgSpeed = (currentSpeed+accelerate*targetSpeed)/2 # m/s
            if avgSpeed <= 0:
                avgSpeed = 0.1
            T = distToLight/avgSpeed  # s (Time to reach target)
            a = dv/dt
        alpha = minJerk(T, 0,currentSpeed,a, distToLight,accelerate*targetSpeed,0)
        setSpeed = [0]*num_waypoints
        if alpha is None: # In case matrix inversion failed
            for i in range(len(setSpeed)):
                if min_index+1 >= len(WPs):
                    setSpeed[i] = self.get_waypoint_velocity(WPs[min_index+i-len(WPs)])
                else:
                    setSpeed[i] = self.get_waypoint_velocity(WPs[min_index+i])
            return setSpeed
        coeff = copy.deepcopy(alpha[6:-8:-1]) # Reverse order
        coeffSpeed = [5,4,3,2,1]*coeff[0:5] # Polynomial for speed
        time = 0
        for i in range(len(setSpeed)):
            if min_index+i > light_index: # If past stop line at light
                setSpeed[i] = accelerate*targetSpeed
            else:
                coeff[-1] = alpha[-1] - self.distance(WPs,min_index,min_index+i)
                roots = np.roots(coeff) # Solve positin polynomial for time
                # Filter out impossible roots - nothing complex nor negative
                if time == 0: # Need this to get started
                    time = 0.0001
                else:
                    roots = np.real(roots[np.isreal(roots)]) # Only keep real roots
                    if np.sum(roots>0) == 0:
                        time = 0.0001 # Choose zero time if all roots negative
                    else:
                        time = np.min(roots[roots>time]) # Choose smallest possible root
                setSpeed[i] = np.polyval(coeffSpeed,time) # Evaluate speed polynomial
        # Car won't start unless first speed is not too close to zero
        if currentSpeed == 0 and accelerate:
            i = 0
            while setSpeed[i] < 2:
                setSpeed[i] = 2
                i += 1
        return setSpeed

    def plan(self, num_waypoints):
        """
        Use ground truth traffic light info to stop at reds and go on greens.
        """
        if (self.position is None) or (self.lights is None):
            return None
        # Get index in the lights array for the upcoming light
        index_to_lights = getClosestForwardLight(self.lights, self.position, self.yaw)
        min_distance = float('inf')
        min_index = copy.copy(self.lastWaypoint)
        index = copy.copy(self.lastWaypoint)
        decreasing = True
        while decreasing:
            waypoint_position = self.base_waypoints[index].pose.pose.position
            distance = position_distance(self.position, waypoint_position)
            if distance < min_distance:
                min_distance = distance
                min_index = index
            else:
                decreasing = False
            index += 1
            if index >= len(self.base_waypoints):
                index = 0
        index = copy.copy(self.lastWaypoint)
        decreasing = True
        light_index = 0
        light_distance = float('inf')
        while decreasing:
            # Need to find closest waypoint for upcoming light position
            waypoint_position = self.base_waypoints[index].pose.pose.position
            d = np.sqrt((waypoint_position.x-self.lights[index_to_lights].pose.pose.position.x)**2 + \
                        (waypoint_position.y-self.lights[index_to_lights].pose.pose.position.y)**2)
            if d < light_distance and light_distance > STOP_EARLY_DIST: # Light positions are directly underneath, stop before it
                light_distance = copy.copy(d)
                light_index = copy.copy(index)
            else:
                decreasing = False
            index += 1
            if index >= len(self.base_waypoints):
                index = 0
        # Check to be sure closest waypoint is not behind us
        head = np.arctan2(self.base_waypoints[min_index].pose.pose.position.y   \
                                                            - self.position.y , \
                          self.base_waypoints[min_index].pose.pose.position.x   \
                                                            - self.position.x)
        if np.abs(head - self.yaw) > np.pi/2:
            min_index += 1
            # Check for wrap-around
            if min_index >= len(self.base_waypoints):
                min_index -= len(self.base_waypoints)
            # Update heading for next check
            head = np.arctan2(self.base_waypoints[min_index].pose.pose.position.y   \
                                                                - self.position.y , \
                              self.base_waypoints[min_index].pose.pose.position.x   \
                                                                - self.position.x)

        WPs = self.base_waypoints # Just temporary as shortcut for variable name
        if min_index > light_index and self.distance(self.base_waypoints,light_index,min_index)<10:
            distToLight = 0
        else:
            distToLight = self.distance(self.base_waypoints,min_index,light_index) # meters
        currentSpeed = self.get_waypoint_velocity(WPs[min_index]) # Current m/s target
        if min_index+decelDist >= len(WPs): # Stop when approaching last waypoint
            light_index = len(WPs) - 5
            distToLight = self.distance(self.base_waypoints,min_index,light_index)
            setSpeed = self.getJMT(currentSpeed, WPs, min_index, light_index, distToLight, num_waypoints, False)
        else:
            if self.lights[index_to_lights].state != 2 and distToLight == 0:
                # Stop for sure if already past desired position
                self.targetWaypoint = None
                setSpeed = [0]*num_waypoints
            else:
                # Check distance to light
                if self.lights[index_to_lights].state != 2 and distToLight < decelDist and not self.initialize:
                    self.targetWaypoint = None
                    setSpeed = self.getJMT(currentSpeed, WPs, min_index, light_index, distToLight, num_waypoints, False)
                else: # Full speed ahead
                    if np.abs(currentSpeed-targetSpeed) < 0.01:
                        setSpeed = [targetSpeed]*num_waypoints
                        self.targetWaypoint = None
                    else:
                        if self.targetWaypoint is None:
                            dist = 0
                            self.targetWaypoint = min_index
                            # Find waypoint index closest to accelDist away
                            while dist < accelDist:
                                self.targetWaypoint += 1
                                if self.targetWaypoint >= len(WPs):
                                    self.targetWaypoint = 0
                                dist = self.distance(WPs,min_index,self.targetWaypoint)
                        dist = self.distance(WPs,min_index,self.targetWaypoint)
                        setSpeed = self.getJMT(currentSpeed, WPs, min_index, self.targetWaypoint, dist, num_waypoints, True)
        #print(setSpeed)
        for i in range(num_waypoints-1): # Loop through all waypoints in list
            if min_index+i >= len(WPs):
                self.set_waypoint_velocity(WPs,min_index+i-len(WPs),setSpeed[i])
            else:
                self.set_waypoint_velocity(WPs,min_index+i         ,setSpeed[i])
        #print("A", min_index, light_index, distToLight, self.get_waypoint_velocity(WPs[min_index]), self.lights[index_to_lights].state, self.position.x,self.position.y)
        self.lastWaypoint = copy.copy(min_index)
        max_index = min_index + num_waypoints
        if max_index >= len(self.base_waypoints):
            max_index -= len(self.base_waypoints)
            WPs = self.base_waypoints[min_index:] + self.base_waypoints[:max_index]
        else:
            WPs = self.base_waypoints[min_index:max_index]
        self.initialize = False
        return WPs
