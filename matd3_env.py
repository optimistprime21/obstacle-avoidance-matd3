import math
import os
import random
import subprocess
import time
from os import path
import gymnasium as gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.3
TIME_DELTA = 0.2


# !is_in_obstacle_position 
# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
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
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, laser_dim):
        
        self.laser_dim = laser_dim

        #self.observation_space = [np.zeros(24) for _ in range(2)]
        self.observation_space = [
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        ]

        #self.action_space = [np.zeros(2) for _ in range(2)]  # linear.x and angular.z per robot
        self.action_space = [gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) for _ in range(2)]


        self.laser_scan_r0 = np.ones(self.laser_dim) * 3.5
        self.laser_scan_r1 = np.ones(self.laser_dim) * 3.5

        self.targets = [False, False]
        self.dones = [False, False]
        self.collisions = [False, False]
        self.donebefore = [False, False]
        self.prev_distances = [0.0, 0.0]

        self.odom_r0 = [0, 0]
        self.odom_r1 = [0, 0]


        self.r0_desired_point = [0,0]
        self.r1_desired_point = [0,0]

        self.upper = 1.75
        self.lower = -1.75

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


        # Initialize ROS node
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")





        # We start all the ROS related Subscribers and publishers
        rospy.Subscriber("r0/odom", Odometry, self.odom_callback_r0)
        rospy.Subscriber("r1/odom", Odometry, self.odom_callback_r1)

        rospy.Subscriber("r0/scan", LaserScan, self.lidar_callback_r0)
        rospy.Subscriber("r1/scan", LaserScan, self.lidar_callback_r1)
        
        self.vel_pub_r0 = rospy.Publisher('r0/cmd_vel', Twist, queue_size=10)
        self.vel_pub_r1= rospy.Publisher('r1/cmd_vel', Twist, queue_size=10)

        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher_r0 = rospy.Publisher("r0_goal_point", MarkerArray, queue_size=3)
        self.publisher_r1 = rospy.Publisher("r1_goal_point", MarkerArray, queue_size=3)



    def odom_callback_r0(self, data):
        self.last_odom_r0 = data

    def odom_callback_r1(self, data):
        self.last_odom_r1 = data





    def lidar_callback_r0(self, msg): 
        max_range = 3.5
        raw_lidar = np.array(msg.ranges)

        # Replace NaNs and inf with max_range
        raw_lidar = np.where(np.isfinite(raw_lidar), raw_lidar, max_range)

        front_half_start = 270  # Start 90° to the right
        front_half_end = 90  # End 90° to the left
        front_lidar = np.concatenate((raw_lidar[front_half_start:], raw_lidar[:front_half_end]))

        # Downsample to 20 values
        bins = np.array_split(front_lidar, 20)
        self.laser_scan_r0 = np.array([np.min(b) for b in bins])  # Take min value per bin



    def lidar_callback_r1(self, msg):
        max_range = 3.5
        raw_lidar = np.array(msg.ranges)

        # Replace NaNs and inf with max_range
        raw_lidar = np.where(np.isfinite(raw_lidar), raw_lidar, max_range)

        front_half_start = 270  # Start 90° to the right
        front_half_end = 90  # End 90° to the left
        front_lidar = np.concatenate((raw_lidar[front_half_start:], raw_lidar[:front_half_end]))

        # Downsample to 20 values
        bins = np.array_split(front_lidar, 20)
        self.laser_scan_r1 = np.array([np.min(b) for b in bins])  # Take min value per bin







    def calculate_distance_and_theta(self, robot_odom, desired_point, angle):
        
        distance = np.linalg.norm(
            [robot_odom[0] - desired_point[0], robot_odom[1] - desired_point[1]]
        )
        skew_x = desired_point[0] - robot_odom[0]
        skew_y = desired_point[1] - robot_odom[1]


        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
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

        quaternion = Quaternion(
            current_position.pose.pose.orientation.w,
            current_position.pose.pose.orientation.x,
            current_position.pose.pose.orientation.y,
            current_position.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm([current_position.pose.pose.position.x - desired_point[0], current_position.pose.pose.position.y - desired_point[1]])

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = desired_point[0] - current_position.pose.pose.position.x
        skew_y = desired_point[1] - current_position.pose.pose.position.y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        return distance, theta
    



    def get_obs(self):
            
        while self.laser_scan_r0 is None or self.laser_scan_r1 is None:
            rospy.sleep(0.1)

        #for robot0:
        odometry = self.last_odom_r0
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        # We round to only two decimals to avoid very big Observation space
        self.odom_r0 = [round(x_position, 2), round(y_position, 2)]

        # for robot1:
        odometry = self.last_odom_r1
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        self.odom_r1 = [round(x_position, 2), round(y_position, 2)]

        observations = []
        observations_r0 = [self.odom_r0, self.laser_scan_r0]
        observations_r1 = [self.odom_r1, self.laser_scan_r1]
        observations = [observations_r0, observations_r1]
        
        return observations
    




    def observe_collision(self, laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, min_laser # collision, min_laser
        return False, min_laser
    

    




    def set_action(self, actions):
        vel_cmd0 = Twist()
        vel_cmd1 = Twist()

        # For robot 0
        if not self.dones[0]:  # Only act if not done
            action = np.array(actions[0])
            raw_linear = action[0]
            raw_angular = action[1]
            linear_vel = (raw_linear + 1.0) / 2.0
            angular_vel = raw_angular
        else:
            linear_vel = 0.0
            angular_vel = 0.0

        vel_cmd0.linear.x = linear_vel
        vel_cmd0.angular.z = angular_vel
        self.vel_pub_r0.publish(vel_cmd0)
        action0 = [linear_vel, angular_vel]

        # For robot 1
        if not self.dones[1]:
            action = np.array(actions[1])
            raw_linear = action[0]
            raw_angular = action[1]
            linear_vel = (raw_linear + 1.0) / 2.0
            angular_vel = raw_angular
        else:
            linear_vel = 0.0
            angular_vel = 0.0

        vel_cmd1.linear.x = linear_vel
        vel_cmd1.angular.z = angular_vel
        self.vel_pub_r1.publish(vel_cmd1)
        action1 = [linear_vel, angular_vel]

        print("Actions = [{}, {}]".format(action0, action1))







    
    def change_goal(self, current_position):
        # Place a new goal and check if its location is not on one of the obstacles
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






    """
    def get_reward(self, target, collision, action, min_laser, donebefore):
        
        if donebefore:
            return 0
        else:
            if target:
                return 100
            elif collision:
                return -100
            else:
                return action[0] / 2 - abs(action[1])  - (1 - min_laser if min_laser < 1 else 0) / 2
    """

    def get_reward(self, donebefore, done, distance, prev_distance, collision, target, action, min_laser):
        
        if donebefore:
            return 0
        else:
            if done:
                if target:
                    return 100.0  # big reward for success

                if collision:
                    return -100.0  # big penalty for crashing
            else:
                # Reward for progress
                #distance_reward = (prev_distance - distance) * 10.0

                # Penalty for getting too close to obstacles
                obstacle_penalty = 0.0
                if min_laser < 1.0:
                    obstacle_penalty = (1.0 - min_laser) 

                # Penalty for sharp turns and high speed (encourage smoothness)
                turning_penalty = 0.5 * abs(action[1])
                forward_reward = 0.5 * action[0]  # reward for moving forward

                # Time penalty to encourage faster solutions
                # time_penalty = 0.01

                reward = forward_reward - obstacle_penalty - turning_penalty
                return reward






    def reset_robot_state(self, robot_state, quaternion):
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

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        # Reset robot 0
        self.reset_robot_state(self.set_r0_state, quaternion)

        # Reset robot 1
        self.reset_robot_state(self.set_r1_state, quaternion)

        self.odom_r0 = [self.set_r0_state.pose.position.x, self.set_r0_state.pose.position.y]
        self.odom_r1 = [self.set_r1_state.pose.position.x ,self.set_r1_state.pose.position.y]


        # Set new desired points for both robots
        self.r0_desired_point = self.change_goal(self.odom_r0)
        self.r1_desired_point = self.change_goal(self.odom_r1)


        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers()



        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")



        # Reset the initial states
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

        # Reset the robot status
        self.targets = [False, False]
        self.dones = [False, False]
        self.collisions = [False, False]
        self.donebefore = [False, False]
        
        return states
    





    # Perform an action and read a new state
    def step(self, actions):
        

        # Publish the robot action
        self.set_action(actions)
        self.publish_markers()

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")


        observations = self.get_obs()
        states = []
        rewards = []

        # Check status for each robot
        for i in range(2):
            laser_data = observations[i][1]
            last_odom = self.last_odom_r0 if i==0 else self.last_odom_r1
            desired_point = self.r0_desired_point if i==0 else self.r1_desired_point

            distance, theta = self.quaternion_distance_and_theta(desired_point, last_odom)

            robot_state = [distance, theta, actions[i][0], actions[i][1]]  
            state = np.append(laser_data, robot_state)
            states.append(state)

            # Check for collision
            collision, min_laser = self.observe_collision(laser_data)


            if collision:
                self.collisions[i] = True

            # Check if the robot reached the desired position
            if distance < GOAL_REACHED_DIST:
                self.targets[i] = True

            # If the robot reached the goal or collided, the episode is done
            if self.targets[i] or self.collisions[i]:
                self.dones[i] = True


            rewards.append(self.get_reward(self.donebefore[i], self.dones[i], distance, self.prev_distances[i], self.collisions[i], self.targets[i], actions[i], min_laser))
            
            print(f"[Robot {i}] min_laser: {min_laser:.2f} | Distance: {distance:.2f} | Collision: {self.collisions[i]} | Target: {self.targets[i]} | Reward: {rewards[i]:.2f}")
            
            self.prev_distances[i] = distance

            if not self.donebefore[i] and self.dones[i]:
                self.donebefore[i] = True
            

        print("Episode dones: ", self.dones)
        return states, rewards, self.dones, self.targets







    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
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
        # Publish visual data in Rviz
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
