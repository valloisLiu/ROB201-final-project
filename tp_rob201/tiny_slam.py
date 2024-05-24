""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        # delete the points that are too far
        mask = distances < np.max(distances)
        distances = distances[mask]
        angles = angles[mask]

        obs_x = distances * np.cos(angles + pose[2]) + pose[0]
        obs_y = distances * np.sin(angles + pose[2]) + pose[1]

        x_px, y_px = self.grid.conv_world_to_map(obs_x, obs_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.grid.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.grid.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        score = np.sum(self.grid.occupancy_map[x_px, y_px])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        corrected_pose = np.zeros(3)

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # # methode 1
        # dis = np.linalg.norm(odom_pose[:2])
        # corrected_pose[0] = odom_pose_ref[0] + dis * np.cos(odom_pose[2] + odom_pose_ref[2])
        # corrected_pose[1] = odom_pose_ref[1] + dis * np.sin(odom_pose[2] + odom_pose_ref[2])
        # corrected_pose[2] = odom_pose[2] + odom_pose_ref[2]

        # methode 2
        corrected_pose[0] = odom_pose_ref[0] + odom_pose[0] * np.cos(odom_pose_ref[2]) - odom_pose[1] * np.sin(odom_pose_ref[2])
        corrected_pose[1] = odom_pose_ref[1] + odom_pose[0] * np.sin(odom_pose_ref[2]) + odom_pose[1] * np.cos(odom_pose_ref[2])
        corrected_pose[2] = odom_pose_ref[2] + odom_pose[2]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = -np.inf
        n = 0
        sigma = 0.01
        
        while ( n < 100 ):
            change = np.random.normal(0, sigma,3)
            change[2] = change[2]
            ODOM_POSE_REF = self.odom_pose_ref + change
            pose = self.get_corrected_pose(raw_odom_pose, ODOM_POSE_REF)
            current_score = self._score(lidar, pose)

            if current_score > best_score:
                best_score = current_score
                n = 0
                self.odom_pose_ref = ODOM_POSE_REF

            n += 1

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        current_position = np.array([pose[0], pose[1]])
        current_angle = pose[2]

        p1, p2 = 0.99, 0.01
        val1, val2 = np.log(p1/(1-p1)), np.log(p2/(1-p2))
        
        obs_x = distances * np.cos(angles + current_angle) + current_position[0]
        obs_y = distances * np.sin(angles + current_angle) + current_position[1]
        self.grid.add_map_points(obs_x, obs_y, val1)

        # prop = 0.9
        # obs_x_line = prop * distances * np.cos(angles + current_angle) + current_position[0]
        # obs_y_line = prop * distances * np.sin(angles + current_angle) + current_position[1]
        dis_mur = 30
        obs_x_line = (distances - dis_mur) * np.cos(angles + current_angle) + current_position[0]
        obs_y_line = (distances - dis_mur) * np.sin(angles + current_angle) + current_position[1]
        for i in range(len(distances)):
            self.grid.add_map_line(current_position[0], current_position[1], obs_x_line[i], obs_y_line[i], val2)
        
        seuil = 8
        self.grid.occupancy_map[self.grid.occupancy_map > seuil] = seuil
        self.grid.occupancy_map[self.grid.occupancy_map < -seuil] = -seuil
        
        self.grid.display2(pose)

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
