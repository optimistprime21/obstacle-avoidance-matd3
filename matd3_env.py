# This code adapts the single-agent TD3 implementation from:
# Reinis Cimurs, Il Hong Suh, Jin Han Lee,
# "Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning,"
# IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 730-737, 2022.
# DOI: 10.1109/LRA.2021.3133591


import math
import os
import random
import subprocess
import time
from os import path
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
import csv



GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.3
TIME_DELTA = 0.2



def check_pos(x, y):
    """
    Check if the position (x, y) is valid (not inside an obstacle or out of bounds).
    """
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok




class GazeboEnv:
    """
    Manages the ROS/Gazebo simulation for multi-agent navigation.
    Handles robot state, sensors, actions, rewards, and environment resets.

    This environment is designed for two robots. It can be reconfigured for different numbers of robots.
    """

    def __init__(self, launchfile, laser_dim):
        # --- Environment and robot state initialization ---
        self.laser_dim = laser_dim
        self.random_near_obstacle = True
        self.count_rand_actions = [0 for _ in range(2)]
        self.random_actions = [np.zeros(2) for _ in range(2)]
        self.laser_scan_r0 = np.ones(self.laser_dim) * 3.5
        self.laser_scan_r1 = np.ones(self.laser_dim) * 3.5
        self.targets = [False, False]
        self.dones = [False, False]
        self.collisions = [False, False]
        self.donebefore = [False, False]
        self.prev_distances = [0.0, 0.0]
        self.all_reward_components = [[], []]
        self.odom_r0 = [0, 0]
        self.odom_r1 = [0, 0]
        self.last_odom_r0 = None
        self.last_odom_r1 = None
        self.r0_desired_point = [0,0]
        self.r1_desired_point = [0,0]
        self.upper = 1.75
        self.lower = -1.75

        # --- Set initial robot states ---
        self.set_r0_state = ModelState()
        self.set_r0_state.model_name = "r0"
        self.set_r0_state.pose.position.x = 0.0
        self.set_r0_state.pose.position.y = 0.0
        self.set_r0_state.pose.position.z = 0.0
        self.set_r0_state.pose.orientation.x = 0.0
        self.set_r0_state.pose.orientation.y = 0.0
        self.set_r0_state.pose.orientation.z = 0.0
        self.set_r0_state.pose.orientation.w = 1.0

        self.set_r1_state = ModelState()
        self.set_r1_state.model_name = "r1"
        self.set_r1_state.pose.position.x = 1.0
        self.set_r1_state.pose.position.y = 1.0
        self.set_r1_state.pose.position.z = 0.0
        self.set_r1_state.pose.orientation.x = 0.0
        self.set_r1_state.pose.orientation.y = 0.0
        self.set_r1_state.pose.orientation.z = 0.0
        self.set_r1_state.pose.orientation.w = 1.0


        # --- Launch ROS core and simulation ---
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")


        # --- ROS Subscribers and Publishers ---
        rospy.Subscriber("r0/odom", Odometry, self.odom_callback_r0)
        rospy.Subscriber("r1/odom", Odometry, self.odom_callback_r1)
        rospy.Subscriber("r0/scan", LaserScan, self.lidar_callback_r0)
        rospy.Subscriber("r1/scan", LaserScan, self.lidar_callback_r1)
        self.vel_pub_r0 = rospy.Publisher('r0/cmd_vel', Twist, queue_size=10)
        self.vel_pub_r1 = rospy.Publisher('r1/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher_r0 = rospy.Publisher("r0_goal_point", MarkerArray, queue_size=3)
        self.publisher_r1 = rospy.Publisher("r1_goal_point", MarkerArray, queue_size=3)



    def odom_callback_r0(self, data):
        """Callback for robot 0 odometry."""
        self.last_odom_r0 = data

    def odom_callback_r1(self, data):
        """Callback for robot 1 odometry."""
        self.last_odom_r1 = data



    def lidar_callback_r0(self, msg): 
        """Process robot 0 lidar scan and downsample to laser_dim bins."""
        max_range = 3.5
        raw_lidar = np.array(msg.ranges)
        raw_lidar = np.where(np.isfinite(raw_lidar), raw_lidar, max_range)
        front_half_start = 270
        front_half_end = 90
        front_lidar = np.concatenate((raw_lidar[front_half_start:], raw_lidar[:front_half_end]))
        bins = np.array_split(front_lidar, 20)
        self.laser_scan_r0 = np.array([np.min(b) for b in bins])

    def lidar_callback_r1(self, msg):
        """Process robot 1 lidar scan and downsample to laser_dim bins."""
        max_range = 3.5
        raw_lidar = np.array(msg.ranges)
        raw_lidar = np.where(np.isfinite(raw_lidar), raw_lidar, max_range)
        front_half_start = 270
        front_half_end = 90
        front_lidar = np.concatenate((raw_lidar[front_half_start:], raw_lidar[:front_half_end]))
        bins = np.array_split(front_lidar, 20)
        self.laser_scan_r1 = np.array([np.min(b) for b in bins])




    def calculate_distance_and_theta(self, robot_odom, desired_point, angle):
        """
        Calculate distance and heading difference between robot and goal.
        """
        distance = np.linalg.norm([robot_odom[0] - desired_point[0], robot_odom[1] - desired_point[1]])
        skew_x = desired_point[0] - robot_odom[0]
        skew_y = desired_point[1] - robot_odom[1]
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = 1.0
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = -np.pi + (theta - np.pi)
        elif theta < -np.pi:
            theta = np.pi + (theta + np.pi)
        return distance, theta



    def quaternion_distance_and_theta(self, desired_point, current_position):
        """
        Calculate distance and heading difference using quaternion orientation.
        """
        quaternion = Quaternion(
            current_position.pose.pose.orientation.w,
            current_position.pose.pose.orientation.x,
            current_position.pose.pose.orientation.y,
            current_position.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        distance = np.linalg.norm([current_position.pose.pose.position.x - desired_point[0], current_position.pose.pose.position.y - desired_point[1]])
        skew_x = desired_point[0] - current_position.pose.pose.position.x
        skew_y = desired_point[1] - current_position.pose.pose.position.y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x**2 + skew_y**2)
        mag2 = 1.0
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        return distance, theta



    def get_obs(self):
        """
        Get current observations for both robots (position and laser scan).
        """
        while self.laser_scan_r0 is None or self.laser_scan_r1 is None or self.last_odom_r0 is None or self.last_odom_r1 is None:
            rospy.sleep(0.01)
        odometry = self.last_odom_r0
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        self.odom_r0 = [round(x_position, 2), round(y_position, 2)]
        odometry = self.last_odom_r1
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        self.odom_r1 = [round(x_position, 2), round(y_position, 2)]
        observations = [
            [self.odom_r0, self.laser_scan_r0],
            [self.odom_r1, self.laser_scan_r1]
        ]
        return observations
    


    def observe_collision(self, laser_data):
        """
        Detect a collision from laser data.
        """
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, min_laser
        return False, min_laser
    


    def set_action(self, actions):
        """
        Publish velocity commands for both robots based on actions.
        Handles random exploration near obstacles.
        """
        vel_cmd0 = Twist()
        vel_cmd1 = Twist()
        for i in range(2):
            if self.dones[i]:
                linear_vel = 0.0
                angular_vel = 0.0
            else:
                action = np.array(actions[i])
                raw_linear = action[0]
                raw_angular = action[1]
                laser_scan = self.laser_scan_r0 if i == 0 else self.laser_scan_r1
                if self.random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(laser_scan) < 0.6
                        and self.count_rand_actions[i] < 1
                    ):
                        self.count_rand_actions[i] = np.random.randint(8, 15)
                        self.random_actions[i] = np.random.uniform(-1, 1, 2)
                    if self.count_rand_actions[i] > 0:
                        self.count_rand_actions[i] -= 1
                        self.random_actions[i][0] = -1
                        raw_linear = self.random_actions[i][0]
                        raw_angular = self.random_actions[i][1]
                linear_vel = (raw_linear + 1.0) / 2.0
                angular_vel = raw_angular
            if i == 0:
                vel_cmd0.linear.x = linear_vel
                vel_cmd0.angular.z = angular_vel
                action0 = [linear_vel, angular_vel]
            else:
                vel_cmd1.linear.x = linear_vel
                vel_cmd1.angular.z = angular_vel
                action1 = [linear_vel, angular_vel]
        self.vel_pub_r0.publish(vel_cmd0)
        self.vel_pub_r1.publish(vel_cmd1)
        print("Actions = [{}, {}]".format(action0, action1))




    def change_goal(self, current_position):
        """
        Randomly select a new goal position for a robot, avoiding obstacles.
        """
        if self.upper < 3.5:
            self.upper += 0.004
        if self.lower > -3.5:
            self.lower -= 0.004
        goal_ok = False
        while not goal_ok:
            x = current_position[0] + random.uniform(self.upper, self.lower)
            y = current_position[1] + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(x, y)
        desired_point = [x, y]
        return desired_point
    


    def get_reward(self, agent_id, evaluate, prev_distance, distance, donebefore, done, collision, target, action, min_laser):
        """
        Calculate the reward for a robot based on its state and action.
        Logs reward components for analysis.
        """
        forward_reward = 0
        turning_penalty = 0
        target_reward = 0
        collision_penalty = 0
        obstacle_penalty = 0
        big_distance_reward = 0
        prev_distance_reward = 0
        if donebefore:
            total_reward = 0
        else:
            if done:
                if target:
                    target_reward = 7000
                if collision:
                    collision_penalty = -3000
            else:
                big_distance_reward = (10 - distance)*2
                prev_distance_reward = (prev_distance - distance)*1500
                obstacle_penalty = 0.0
                if min_laser < 0.5:
                    obstacle_penalty = (1.0 - min_laser)*200
                turning_penalty = 2* abs(action[1])
                forward_reward = 2* action[0]
            total_reward = big_distance_reward + prev_distance_reward + forward_reward - turning_penalty - obstacle_penalty + target_reward + collision_penalty
        reward_components = {
            "agent_id" : agent_id,
            "big_distance_reward": big_distance_reward,
            "prev_distance_reward": prev_distance_reward,
            "forward_reward": forward_reward,
            "turning_penalty": turning_penalty,
            "obstacle_penalty": obstacle_penalty,
            "target_reward": target_reward,
            "collision_penalty": collision_penalty,
            "total_reward": total_reward
        }
        if not evaluate:
            self.all_reward_components[agent_id].append(reward_components)
        return total_reward
    



    def reset_robot_state(self, robot_state, quaternion):
        """
        Randomly reset a robot's position and orientation, avoiding obstacles.
        """
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        robot_state.pose.position.x = x
        robot_state.pose.position.y = y
        robot_state.pose.orientation.x = quaternion.x
        robot_state.pose.orientation.y = quaternion.y
        robot_state.pose.orientation.z = quaternion.z
        robot_state.pose.orientation.w = quaternion.w
        self.set_state.publish(robot_state) 




    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Also resets robots, goals, and randomizes obstacles.
        """
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.reset_robot_state(self.set_r0_state, quaternion)
        self.reset_robot_state(self.set_r1_state, quaternion)
        self.odom_r0 = [self.set_r0_state.pose.position.x, self.set_r0_state.pose.position.y]
        self.odom_r1 = [self.set_r1_state.pose.position.x ,self.set_r1_state.pose.position.y]
        self.r0_desired_point = self.change_goal(self.odom_r0)
        self.r1_desired_point = self.change_goal(self.odom_r1)
        self.random_box()
        self.publish_markers()
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        states = []
        for i, odom, desired_point, laser_scan in zip(
            [0, 1],
            [self.odom_r0, self.odom_r1],
            [self.r0_desired_point, self.r1_desired_point],
            [self.laser_scan_r0, self.laser_scan_r1]
        ):
            distance, theta = self.calculate_distance_and_theta(odom, desired_point, angle)
            self.prev_distances[i] = distance
            robot_state = [distance, theta, 0.0, 0.0]  
            state = np.append(laser_scan, robot_state)
            states.append(state)
        self.targets = [False, False]
        self.dones = [False, False]
        self.collisions = [False, False]
        self.donebefore = [False, False]
        return states




    def step(self, actions, evaluate):
        """
        Perform an action and read a new state.
        Returns next states, rewards, done flags, and target flags.
        """
        self.set_action(actions)
        self.publish_markers()
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        observations = self.get_obs()
        states = []
        rewards = []
        for i in range(2):
            laser_data = observations[i][1]
            last_odom = self.last_odom_r0 if i==0 else self.last_odom_r1
            desired_point = self.r0_desired_point if i==0 else self.r1_desired_point
            distance, theta = self.quaternion_distance_and_theta(desired_point, last_odom)
            robot_state = [distance, theta, actions[i][0], actions[i][1]]  
            state = np.append(laser_data, robot_state)
            states.append(state)
            collision, min_laser = self.observe_collision(laser_data)
            if collision:
                self.collisions[i] = True
            if distance < GOAL_REACHED_DIST:
                self.targets[i] = True
            if self.targets[i] or self.collisions[i]:
                self.dones[i] = True
            rewards.append(self.get_reward(i, evaluate, self.prev_distances[i], distance, self.donebefore[i], self.dones[i], self.collisions[i], self.targets[i], actions[i], min_laser))
            print(f"[Robot {i}] min_laser: {min_laser:.2f} | Distance: {distance:.2f} | Collision: {self.collisions[i]} | Target: {self.targets[i]} | Reward: {rewards[i]:.2f}")
            self.prev_distances[i] = distance
            if not self.donebefore[i] and self.dones[i]:
                self.donebefore[i] = True
        print("Episode dones: ", self.dones)
        return states, rewards, self.dones, self.targets





    def random_box(self):
        """
        Randomly change the location of the boxes in the environment on each reset to randomize the training environment.
        """
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_r0 = np.linalg.norm([x - self.odom_r0[0], y - self.odom_r0[1]])
                distance_to_r1 = np.linalg.norm([x - self.odom_r1[0], y - self.odom_r1[1]])
                distance_to_goal0 = np.linalg.norm([x - self.r0_desired_point[0], y - self.r0_desired_point[1]])
                distance_to_goal1 = np.linalg.norm([x - self.r1_desired_point[0], y - self.r1_desired_point[1]])
                if distance_to_r0 < 1.5 or distance_to_goal0 < 1.5 or distance_to_r1 < 1.5 or distance_to_goal1 < 1.5:
                    box_ok = False

            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)




    def publish_markers(self):
        """
        Publish goal markers for both robots in Rviz for visualization.
        """
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "r0_tf/odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.r0_desired_point[0]
        marker.pose.position.y = self.r0_desired_point[1]
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher_r0.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "r1_tf/odom"
        marker2.type = marker.CYLINDER
        marker2.action = marker.ADD
        marker2.scale.x = 0.1
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 0.0
        marker2.color.b = 1.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = self.r1_desired_point[0]
        marker2.pose.position.y = self.r1_desired_point[1]
        marker2.pose.position.z = 0
        markerArray2.markers.append(marker2)
        self.publisher_r1.publish(markerArray2)





    def export_rewards_to_csv(self, output_dir="./reward_logs"):
        """
        Export reward components for each agent to CSV for analysis.
        """
        os.makedirs(output_dir, exist_ok=True)
        for agent_id in range(2):
            filename = os.path.join(output_dir, f"agent_{agent_id}_rewards.csv")
            with open(filename, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "big_distance_reward", "prev_distance_reward", "forward_reward", "turning_penalty", 
                    "obstacle_penalty", "target_reward", "collision_penalty", "total_reward"
                ])
                if file.tell() == 0:
                    writer.writeheader()
                for entry in self.all_reward_components[agent_id]:
                    writer.writerow({
                        "big_distance_reward": entry["big_distance_reward"],
                        "prev_distance_reward": entry["prev_distance_reward"],
                        "forward_reward": entry["forward_reward"],
                        "turning_penalty": entry["turning_penalty"],
                        "obstacle_penalty": entry["obstacle_penalty"],
                        "target_reward": entry["target_reward"],
                        "collision_penalty": entry["collision_penalty"],
                        "total_reward": entry["total_reward"]
                    })
        # Clear after exporting
        self.all_reward_components = [[], []]
