""" A set of robotics control functions """

import random
import numpy as np
import math


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    lidar_data = lidar
    value = np.min(lidar_data.get_sensor_values())
    index = np.argmin(lidar_data.get_sensor_values())
    angle = lidar_data.get_ray_angles()[index]

    distance_min = 50
    forward_speed = 0.5

    # eviter de tourner face a un mur
    if 0 <= angle and angle < np.pi/2 and value < distance_min:
        rotation_speed = np.random.uniform(-1, 0)
        command = {"forward": 0, "rotation": rotation_speed}
    elif 0 > angle and angle > -np.pi/2 and value < distance_min:
        rotation_speed = np.random.uniform(0, 1)
        command = {"forward": 0, "rotation": rotation_speed}
    else:
        command = {"forward": forward_speed, "rotation": 0}

    # # rotation aleatoire
    # if -np.pi/2 < angle and angle < np.pi/2 and value < distance_min:
    #     rotation_speed = np.random.uniform(-1, 1)
    #     command = {"forward": 0, "rotation": rotation_speed}
    # else:
    #     command = {"forward": forward_speed, "rotation": 0}

    return command


def potential_field_control(lidar, current_pose, goal_pose, target):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """ 
    # TODO for TP2
    num_point = len(goal_pose)
    goal_pose = goal_pose[target]

    # system parameters
    d_change = 40
    stop_dist = 5
    d_safe = 40
    K_cone = 0.2
    K_quad = K_cone/d_change
    
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    current_position = np.array([current_pose[0], current_pose[1]])
    current_angle = current_pose[2]
    goal_position = np.array([goal_pose[0], goal_pose[1]])
    
    # gradient
    direction = goal_position - current_position
    distance = np.linalg.norm(direction)

    # l'obstacle le plus proche
    index = np.argmin(distances)
    min_dist = distances[index]
    min_angle = angles[index]
    obstacle_position = np.array([current_pose[0] + min_dist*np.cos(min_angle+current_angle), current_pose[1] + min_dist*np.sin(min_angle+current_angle)])
    
    # eviter des obstacles
    if min_dist < d_safe :
        K_obs = 1000
        gradient_obstacle = K_obs/(min_dist**3)*((1/min_dist)-(1/d_safe))*(obstacle_position - current_position)
    else :
        gradient_obstacle = np.array([0,0])
    
    # Potentiel conique
    if distance > d_change :
        gradient = K_cone/distance*direction
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        speed = np.clip(gradient_norme*np.cos(gradient_angle-current_angle), -1, 1)
        rotation = np.clip(5*gradient_norme*np.sin(gradient_angle-current_angle), -1, 1)
    
    # Potentiel quadratique
    elif stop_dist < distance <= d_change :
        gradient = K_quad*direction
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        speed = np.clip(gradient_norme*np.cos(gradient_angle-current_angle), -1, 1)
        rotation = np.clip(5*gradient_norme*np.sin(gradient_angle-current_angle), -1, 1)
    
    elif distance <= stop_dist :
        speed = 0
        rotation = 0
        if target < num_point-1:
            target += 1
    
    command = {"forward": speed, "rotation": rotation}

    return [command, target]

def potential_field_control_tp5(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """ 
    # TODO for TP2
    num_point = len(goal_pose)

    # system parameters
    d_change = 40
    stop_dist = 5
    d_safe = 40
    K_cone = 0.2
    K_quad = K_cone/d_change
    
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    current_position = np.array([current_pose[0], current_pose[1]])
    current_angle = current_pose[2]
    goal_position = np.array([goal_pose[0], goal_pose[1]])
    
    # gradient
    direction = goal_position - current_position
    distance = np.linalg.norm(direction)

    # l'obstacle le plus proche
    index = np.argmin(distances)
    min_dist = distances[index]
    min_angle = angles[index]
    obstacle_position = np.array([current_pose[0] + min_dist*np.cos(min_angle+current_angle), current_pose[1] + min_dist*np.sin(min_angle+current_angle)])
    
    # eviter des obstacles
    if min_dist < d_safe :
        K_obs = 1000
        gradient_obstacle = K_obs/(min_dist**3)*((1/min_dist)-(1/d_safe))*(obstacle_position - current_position)
    else :
        gradient_obstacle = np.array([0,0])
    
    # Potentiel conique
    if distance > d_change :
        gradient = K_cone/distance*direction
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        speed = np.clip(gradient_norme*np.cos(gradient_angle-current_angle), -1, 1)
        rotation = np.clip(5*gradient_norme*np.sin(gradient_angle-current_angle), -1, 1)
    
    # Potentiel quadratique
    elif stop_dist < distance <= d_change :
        gradient = K_quad*direction
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        speed = np.clip(gradient_norme*np.cos(gradient_angle-current_angle), -1, 1)
        rotation = np.clip(5*gradient_norme*np.sin(gradient_angle-current_angle), -1, 1)
    
    elif distance <= stop_dist :
        speed = 0
        rotation = 0
    
    command = {"forward": speed, "rotation": rotation}

    return command