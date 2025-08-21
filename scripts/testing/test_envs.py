import math
import os
import csv
import subprocess
import time
from os import path
import numpy as np
import rospy
import random
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.3
TIME_DELTA = 0.2

# ---------------------- Utility functions ----------------------

def check_pos_gate(x, y):
    """
    For Gate World - Check if the position (x, y) is valid (not inside an obstacle or out of bounds).
    """
    goal_ok = True

    if x > 2.3 or x < -2.3 or y > 2.3 or y < -2.3:
        goal_ok = False

    return goal_ok

def check_pos_complex(x, y):
    """
    For Complex World - Check if the position (x, y) is valid (not inside an obstacle or out of bounds).
    """
    goal_ok = True

    if x < -0.5 and x > -3.5 and y > 0.5 and y < 3.5:
        goal_ok = False

    if x > 0.5 and x < 3.5 and y > 2 and y < 3.5:
        goal_ok = False

    if x > 2 and x < 3.5 and y > 0.75 and y < 3:
        goal_ok = False

    if x > -3.5 and x < 3.5 and y > -1.5 and y < -0.5:
        goal_ok = False

    if x > -3.5 and x < -1.0 and y > -5.5 and y < -2.6:
        goal_ok = False

    if x > 1.0 and x < 3.0 and y > -5.0 and y < -3.0:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y < -6.6 or y > 4.5:
        goal_ok = False

    return goal_ok

# ---------------------- Base Environment ----------------------

class BaseEnv:
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim

        # States
        self.odom_r0, self.odom_r1 = [0, 0], [0, 0]
        self.goal_r0, self.goal_r1 = [0, 0], [0, 0]

        self.lidar_data_r0 = np.ones(environment_dim) * 3.5
        self.lidar_data_r1 = np.ones(environment_dim) * 3.5

        self.last_odom_r0, self.last_odom_r1 = None, None
        self.targets, self.dones = [False, False], [False, False]
        self.collisions, self.donebefore = [False, False], [False, False]
        self.prev_distances = [0.0, 0.0]
        self.all_reward_components = [[], []]

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

        # ROS init
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        rospy.init_node("gym", anonymous=True)

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "launch", launchfile)
        if not path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        subprocess.Popen(["roslaunch", fullpath])
        print("Gazebo launched!")

        # Publishers / Subscribers
        self.vel_pub_r0 = rospy.Publisher('r0/cmd_vel', Twist, queue_size=10)
        self.vel_pub_r1 = rospy.Publisher('r1/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.publisher_r0 = rospy.Publisher("r0_goal_point", MarkerArray, queue_size=3)
        self.publisher_r1 = rospy.Publisher("r1_goal_point", MarkerArray, queue_size=3)

        rospy.Subscriber("r0/scan", LaserScan, self.lidar_callback_r0)
        rospy.Subscriber("r1/scan", LaserScan, self.lidar_callback_r1)
        rospy.Subscriber("r0/odom", Odometry, self.odom_callback_r0)
        rospy.Subscriber("r1/odom", Odometry, self.odom_callback_r1)

    # --------- Callbacks ---------
    def lidar_callback_r0(self, msg):
        self.lidar_data_r0 = self._process_lidar(msg)

    def lidar_callback_r1(self, msg):
        self.lidar_data_r1 = self._process_lidar(msg)

    def odom_callback_r0(self, data):
        self.last_odom_r0 = data

    def odom_callback_r1(self, data):
        self.last_odom_r1 = data

    def _process_lidar(self, msg):
        max_range = 3.5
        raw_lidar = np.where(np.isfinite(msg.ranges), msg.ranges, max_range)
        front_half = np.concatenate((raw_lidar[270:], raw_lidar[:90]))
        bins = np.array_split(front_half, 20)
        return np.array([np.min(b) for b in bins])

    # --------- Actions and Observations ---------


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


    def is_far_enough(self, point, others, min_dist=1):
        """Ensure 'point' is at least min_dist away from all 'others'."""
        for other in others:
            if np.linalg.norm(np.array(point) - np.array(other)) < min_dist:
                return False
        return True


    def get_obs(self):

        while self.lidar_data_r0 is None or self.lidar_data_r1 is None:
            rospy.sleep(0.1)

        odoms = [self.last_odom_r0, self.last_odom_r1]
        obs = []
        for i, (odom, lidar) in enumerate(zip(odoms, [self.lidar_data_r0, self.lidar_data_r1])):
            x, y = odom.pose.pose.position.x, odom.pose.pose.position.y
            if i == 0:
                self.odom_r0 = [round(x, 2), round(y, 2)]
            else:
                self.odom_r1 = [round(x, 2), round(y, 2)]
            obs.append([[x, y], lidar])
        return obs

    # --------- Reward helpers ---------
    def observe_collision(self, laser_data):
        min_laser = min(laser_data)
        return min_laser < COLLISION_DIST, min_laser

    def calculate_distance_and_theta(self, robot_odom, goal_point, angle):
        
        distance = np.linalg.norm(
            [robot_odom[0] - goal_point[0], robot_odom[1] - goal_point[1]]
        )
        skew_x = goal_point[0] - robot_odom[0]
        skew_y = goal_point[1] - robot_odom[1]


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
    

    def quaternion_distance_and_theta(self, goal_point, current_position):

        quaternion = Quaternion(
            current_position.pose.pose.orientation.w,
            current_position.pose.pose.orientation.x,
            current_position.pose.pose.orientation.y,
            current_position.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm([current_position.pose.pose.position.x - goal_point[0], current_position.pose.pose.position.y - goal_point[1]])

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = goal_point[0] - current_position.pose.pose.position.x
        skew_y = goal_point[1] - current_position.pose.pose.position.y
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
    
    def get_reward(self, agent_id, evaluate, prev_distance, distance, donebefore, done, collision, target, action, min_laser):
        
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
                
                big_distance_reward = (15 - distance)*2
                prev_distance_reward = (prev_distance - distance)*1500
                
                # Penalty for getting too close to obstacles
                obstacle_penalty = 0.0
                if min_laser < 1.0:
                    obstacle_penalty = (1.0 - min_laser)*200

                # Penalty for sharp turns
                turning_penalty = 10* abs(action[1])
                forward_reward = 10* action[0]  # reward for moving forward

            total_reward = big_distance_reward + prev_distance_reward + forward_reward - turning_penalty - obstacle_penalty + target_reward + collision_penalty
            
        reward_components = {
            "agent_id" : agent_id,
            "big_distance_reward" : big_distance_reward,
            "prev_distance_reward" : prev_distance_reward,
            "forward_reward": forward_reward,
            "turning_penalty": turning_penalty,
            "obstacle_penalty": obstacle_penalty,
            "target_reward": target_reward,
            "collision_penalty": collision_penalty,
            "total_reward": total_reward
        }

        if evaluate == False:
            self.all_reward_components[agent_id].append(reward_components)
            
        return total_reward

    # --------- Common step/reset/markers/export ---------

    def step(self, actions):

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
        states, rewards = [], []
        for i in range(2):
            laser_data = observations[i][1]
            last_odom = self.last_odom_r0 if i == 0 else self.last_odom_r1
            goal_point = self.goal_r0 if i == 0 else self.goal_r1
            distance, theta = self.quaternion_distance_and_theta(goal_point, last_odom)

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

            reward = self.get_reward(i, False, self.prev_distances[i], distance, self.donebefore[i], self.dones[i], self.collisions[i], self.targets[i], actions[i], min_laser)
            rewards.append(reward)
            print(f"[Robot {i}] min_laser: {min_laser:.2f} | Distance: {distance:.2f} | Collision: {self.collisions[i]} | Target: {self.targets[i]} | Reward: {rewards[i]:.2f}")

            self.prev_distances[i] = distance

            if not self.donebefore[i] and self.dones[i]:
                self.donebefore[i] = True

        print("Episode dones: ", self.dones)

        return states, rewards, self.dones, self.targets





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
        marker.pose.position.x = self.goal_r0[0]
        marker.pose.position.y = self.goal_r0[1]
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
        marker2.pose.position.x = self.goal_r1[0]
        marker2.pose.position.y = self.goal_r1[1]
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher_r1.publish(markerArray2)


    def export_rewards_to_csv(self, output_dir="./test_reward_logs"):
        os.makedirs(output_dir, exist_ok=True)
        for agent_id in range(2):
            filename = os.path.join(output_dir, f"agent_{agent_id}_rewards.csv")
            with open(filename, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "forward_reward", "big_distance_reward",
                    "prev_distance_reward", "turning_penalty", "obstacle_penalty",
                    "target_reward", "collision_penalty", "total_reward"
                ])
                if file.tell() == 0:
                    writer.writeheader()
                for entry in self.all_reward_components[agent_id]:
                    writer.writerow(entry)
        self.all_reward_components = [[], []]

    def reset(self):
        raise NotImplementedError
    
    def reset_robot_state(self, *args, **kwargs):
        raise NotImplementedError

# ---------------------- ComplexEnv ----------------------

class ComplexEnv(BaseEnv):

    def __init__(self, launchfile, environment_dim):
        super().__init__(launchfile, environment_dim)
        self.upper, self.lower = 3.0, -3.0
        self.goal_r0, self.goal_r1 = [2, 2], [-2, -2]


    def reset_robot_state(self, robot_state, quaternion, occupied_positions):
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-7, 4.5)
            position_ok = check_pos_complex(x, y) and self.is_far_enough((x, y), occupied_positions)
        robot_state.pose.position.x = x
        robot_state.pose.position.y = y
        robot_state.pose.orientation.x = quaternion.x
        robot_state.pose.orientation.y = quaternion.y
        robot_state.pose.orientation.z = quaternion.z
        robot_state.pose.orientation.w = quaternion.w
        self.set_state.publish(robot_state)
        return (x, y)
    
    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")


        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        

        occupied_positions = []

        # Reset robot 0
        pos_r0 = self.reset_robot_state(self.set_r0_state, quaternion, occupied_positions)
        occupied_positions.append(pos_r0)

        # Reset robot 1
        pos_r1 = self.reset_robot_state(self.set_r1_state, quaternion, occupied_positions)
        occupied_positions.append(pos_r1)

        self.odom_r0 = list(pos_r0)
        self.odom_r1 = list(pos_r1)

        # Set new goal points
        goal_r0 = self.change_goal(self.odom_r0, occupied_positions)
        occupied_positions.append(goal_r0)

        goal_r1 = self.change_goal(self.odom_r1, occupied_positions)
        occupied_positions.append(goal_r1)

        self.goal_r0 = goal_r0
        self.goal_r1 = goal_r1

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

        for i, odom, goal_point, laser_scan in zip(
            [0, 1],
            [self.odom_r0, self.odom_r1],
            [self.goal_r0, self.goal_r1],
            [self.lidar_data_r0, self.lidar_data_r1]
        ):
            distance, theta = self.calculate_distance_and_theta(odom, goal_point, angle)
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


# ---------------------- GateEnv ----------------------

class GateEnv(BaseEnv):

    def __init__(self, launchfile, environment_dim):
        super().__init__(launchfile, environment_dim)
        self.upper, self.lower = 1.75, -1.75
        self.goal_r0, self.goal_r1 = [-1, 3.2], [1, 3.2]


    def reset_robot_state(self, x, y, quat, robot_state):
            robot_state.pose.position.x = x
            robot_state.pose.position.y = y
            robot_state.pose.orientation.x = quat.x
            robot_state.pose.orientation.y = quat.y
            robot_state.pose.orientation.z = quat.z
            robot_state.pose.orientation.w = quat.w
            self.set_state.publish(robot_state)

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        
        angle = math.pi / 2  # 90 degrees
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        
        
        # Reset robot 0
        self.reset_robot_state(-1, 0.5, quaternion, self.set_r0_state)

        # Reset robot 1
        self.reset_robot_state(1, 0.5, quaternion, self.set_r1_state)


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

        for i, odom, goal_point, laser_scan in zip(
            [0, 1],
            [self.odom_r0, self.odom_r1],
            [self.goal_r0, self.goal_r1],
            [self.lidar_data_r0, self.lidar_data_r1]
        ):
            distance, theta = self.calculate_distance_and_theta(odom, goal_point, angle)
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