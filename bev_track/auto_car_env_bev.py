import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import math
import threading
import random
from gazebo_msgs.srv import SetEntityState
from core.base_controller import BaseController


class BEVCarEnv(gym.Env):
    def __init__(self):
        super(BEVCarEnv, self).__init__()

        rclpy.init()
        self.node = rclpy.create_node('rl_bev_env_node')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        self.set_state_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('等待 Gazebo 传送服务启动...')

        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.node.create_subscription(ModelStates, '/gazebo/model_states', self.models_callback, 10)
        # 👑 【核心修改】：订阅激光雷达，放弃图像
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.robot_name = 'turtlebot3_waffle_pi'
        self.car_x, self.car_y, self.car_yaw = 0.2, 0.0, 0.0
        self.pedestrian_x, self.pedestrian_y = 100.0, 100.0
        self.barrel_x, self.barrel_y = 2.0, 0.0

        # 👑 【核心修改】：状态变为 64x64 的单通道黑白 BEV 图像
        self.current_bev = np.zeros((64, 64, 1), dtype=np.uint8)

        self.prev_x = 0.2
        self.step_count = 0
        self.max_steps = 200
        self.episode_reward = 0.0

        self.base_controller = BaseController(k_y=0.5, k_yaw=1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观察空间缩小为 64x64 单通道，极大降低算力消耗
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

        self.executor_thread = threading.Thread(target=self.spin_ros, daemon=True)
        self.executor_thread.start()
        time.sleep(2.0)

    def spin_ros(self):
        rclpy.spin(self.node)

    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.car_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def models_callback(self, msg):
        try:
            for i, name in enumerate(msg.name):
                if 'dynamic_pedestrian' in name:
                    self.pedestrian_x = msg.pose[i].position.x
                    self.pedestrian_y = msg.pose[i].position.y
                elif 'waffle' in name or 'turtlebot' in name:
                    self.robot_name = name
        except Exception:
            pass

            # 👑 【极其漂亮的数据管道】：将 1D 雷达变成 2D 上帝视角图像

    def scan_callback(self, msg):
        bev_image = np.zeros((64, 64, 1), dtype=np.uint8)
        center = 32
        max_dist = 3.0
        pixels_per_meter = 32 / max_dist  # 32 像素代表 3 米

        # 处理雷达的 inf 和 NaN 噪音
        ranges = np.nan_to_num(msg.ranges, posinf=max_dist, neginf=0.0)
        ranges = np.clip(ranges, 0, max_dist)

        for i, dist in enumerate(ranges):
            # 忽略太近的噪点（自身车体）和超出范围的点
            if 0.15 < dist < max_dist:
                angle = msg.angle_min + i * msg.angle_increment
                # 前方对应图像上方 (Y轴减小)，左侧对应图像左方 (X轴减小)
                img_y = int(center - (dist * math.cos(angle)) * pixels_per_meter)
                img_x = int(center - (dist * math.sin(angle)) * pixels_per_meter)

                if 0 <= img_x < 64 and 0 <= img_y < 64:
                    bev_image[img_y, img_x, 0] = 255  # 画出极其耀眼的白点

        self.current_bev = bev_image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cmd_vel_pub.publish(Twist())

        req_car = SetEntityState.Request()
        req_car.state.name = self.robot_name
        req_car.state.pose.position.x = 0.2
        req_car.state.pose.position.y = 0.0
        req_car.state.pose.position.z = 0.05
        req_car.state.pose.orientation.w = 1.0
        self.set_state_client.call_async(req_car)

        random_barrel_y = random.uniform(-0.35, 0.35)
        req_barrel = SetEntityState.Request()
        req_barrel.state.name = 'static_barrel'
        req_barrel.state.pose.position.x = 2.0
        req_barrel.state.pose.position.y = random_barrel_y
        req_barrel.state.pose.position.z = 0.035
        req_barrel.state.pose.orientation.w = 1.0
        self.set_state_client.call_async(req_barrel)
        self.barrel_y = random_barrel_y

        time.sleep(1.0)
        self.car_x, self.car_y, self.prev_x = 0.2, 0.0, 0.2
        self.step_count = 0
        self.episode_reward = 0.0

        return self.current_bev, {}

    def step(self, action):
        self.step_count += 1
        dist_barrel = math.hypot(self.car_x - self.barrel_x, self.car_y - self.barrel_y)
        dist_pedestrian = math.hypot(self.car_x - self.pedestrian_x, self.car_y - self.pedestrian_y)

        v_base, theta_base = self.base_controller.get_base_action(self.car_y, self.car_yaw, dist_barrel,
                                                                  dist_pedestrian)
        delta_v = float(np.clip(action[0], -1.0, 1.0)) * 0.1
        delta_theta = float(np.clip(action[1], -1.0, 1.0)) * 1.2

        final_throttle = float(np.clip(v_base + delta_v, 0.0, 0.4))
        final_steering = float(np.clip(theta_base + delta_theta, -1.5, 1.5))

        twist = Twist()
        twist.linear.x, twist.angular.z = final_throttle, final_steering
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.033)

        done = False
        truncated = False
        reward = 0.0

        delta_x = self.car_x - self.prev_x
        self.prev_x = self.car_x

        base_income = delta_x * 50.0
        discount = abs(self.car_y) * 1.0 + abs(delta_theta) * 0.5
        step_reward = max(0.0, base_income - discount) + 0.01

        reward += step_reward
        self.episode_reward += reward

        if self.car_x > 4.8:
            reward += 100.0
            done = True
        elif abs(self.car_y) > 0.6:
            reward -= 100.0
            done = True
        elif dist_barrel < 0.30 or dist_pedestrian < 0.35:
            reward -= 100.0
            done = True

        if self.step_count >= self.max_steps and not done:
            truncated = True

        return self.current_bev, reward, done, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()