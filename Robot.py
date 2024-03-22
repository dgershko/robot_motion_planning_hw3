import itertools
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString
from typing import List, Tuple

class Robot(object):
    
    def __init__(self):

        # define robot properties
        self.links = np.array([80.0,70.0,40.0,40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi/3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        return np.linalg.norm(np.array(prev_config) - np.array(next_config))

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        positions = [(0,0)]
        abs_angle = 0
        for index, angle in enumerate(given_config):
            prev_x, prev_y = positions[index]
            abs_angle += angle
            if abs_angle > np.pi:
                abs_angle -= 2 * np.pi
            elif abs_angle < -np.pi:
                abs_angle += 2 * np.pi
            x = self.links[index] * np.cos(abs_angle)
            y = self.links[index] * np.sin(abs_angle)
            positions.append((prev_x + x, prev_y + y))
        return positions[1:]

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1,len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        total_angle = link_angle + given_angle
        if total_angle > np.pi:
            return total_angle - 2*np.pi
        elif total_angle < -np.pi:
            return total_angle + 2*np.pi
        else:
            return total_angle
        
    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        links = [] # type: List[LineString]
        for i in range(len(robot_positions) - 1):
            links.append(LineString((robot_positions[i], robot_positions[i+1])))
        
        for i in range(len(links) - 2):
            if any([links[i].intersects(other_link) for other_link in links[i + 2:]]):
                return False
        return True
        