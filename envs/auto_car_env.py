import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import math
import threading
import random
from gazebo_msgs.srv import SetEntityState

class AutonomousCarEnv(gym.Env):
    def __init__(self):
        super(AutonomousCarEnv, self).__init__()
        
        # --- 1. 初始化 ROS 2 节点与通信 ---
        rclpy.init()
        self.node = rclpy.create_node('rl_env_node')
        self.bridge = CvBridge()
        
        # 发布控制指令
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        
        # 上帝之手客户端：用于重置位置
        self.set_state_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('等待 Gazebo 传送服务启动...')
        
        # 订阅传感器数据
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.node.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.node.create_subscription(ModelStates, '/gazebo/model_states', self.models_callback, 10)
        
        # --- 2. 状态与变量初始化 ---
        self.robot_name = 'turtlebot3_waffle_pi'  # 默认名，将被动态覆盖
        self.car_x = 0.2  
        self.car_y = 0.0
        self.pedestrian_x = 100.0 
        self.pedestrian_y = 100.0
        self.barrel_x = 2.0       
        self.barrel_y = 0.0       
        
        # 升级为 128x128 的 RGB 三通道彩色矩阵
        self.current_image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        self.prev_x = 0.2
        self.step_count = 0
        self.max_steps = 200      # 20秒物理时间跑完 5 米
        self.episode_reward = 0.0 # 实时计分板
        
        # --- 3. 强化学习空间定义 ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观察空间同步升级为 (128, 128, 3) 格式，SB3 会自动适配处理
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self.executor_thread = threading.Thread(target=self.spin_ros, daemon=True)
        self.executor_thread.start()
        time.sleep(2.0) # 等待数据灌入

    def spin_ros(self):
        rclpy.spin(self.node)

    # --- 回调函数区 ---
    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

    def image_callback(self, msg):
        try:
            # 提取 bgr8 彩色信息，并放大到 128x128
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv2.resize(cv_image, (128, 128))
        except Exception as e:
            self.node.get_logger().error(f"Image Error: {e}")

    def models_callback(self, msg):
        """动态抓取小车和行人的真实名字与坐标"""
        try:
            for i, name in enumerate(msg.name):
                if 'dynamic_pedestrian' in name:
                    self.pedestrian_x = msg.pose[i].position.x
                    self.pedestrian_y = msg.pose[i].position.y
                elif 'waffle' in name or 'turtlebot' in name:
                    self.robot_name = name
        except Exception:
            pass 

    # --- 强化学习核心接口 ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 刹车
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        
        # 2. 传送小车 (动态身份识别)
        req_car = SetEntityState.Request()
        req_car.state.name = self.robot_name  
        req_car.state.pose.position.x = 0.2
        req_car.state.pose.position.y = 0.0
        req_car.state.pose.position.z = 0.05
        req_car.state.pose.orientation.w = 1.0
        self.set_state_client.call_async(req_car)
        
        # 3. 域随机化：随机传送静态铁桶位置防作弊
        random_barrel_y = random.uniform(-0.35, 0.35)
        req_barrel = SetEntityState.Request()
        req_barrel.state.name = 'static_barrel'
        req_barrel.state.pose.position.x = 2.0  
        req_barrel.state.pose.position.y = random_barrel_y
        req_barrel.state.pose.position.z = 0.035
        req_barrel.state.pose.orientation.w = 1.0
        self.set_state_client.call_async(req_barrel)
        
        # 同步更新本地坐标
        self.barrel_y = random_barrel_y

        self.node.get_logger().info(f"【环境】回合重置，铁桶随机刷新至 Y={random_barrel_y:.2f}...")
        
        # 等待物理引擎稳定
        time.sleep(1.0) 
        
        # 4. 内部状态初始化
        self.car_x = 0.2
        self.car_y = 0.0
        self.prev_x = 0.2
        self.step_count = 0
        self.episode_reward = 0.0 # 清空计分板
        
        obs = self.current_image
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        
        # 映射并发布速度
        throttle = float(np.clip(action[0], -1.0, 1.0) + 1.0) / 2.0 * 0.4
        steering = float(np.clip(action[1], -1.0, 1.0)) * 1.0
        
        twist = Twist()
        twist.linear.x = throttle
        twist.angular.z = steering
        self.cmd_vel_pub.publish(twist)
        
        # 同步 Gazebo 的 3倍速，Python 休眠时间降至 0.033
        time.sleep(0.033)
        
        done = False
        truncated = False
        reward = 0.0
        
        delta_x = self.car_x - self.prev_x
        self.prev_x = self.car_x
        
        dist_barrel = math.hypot(self.car_x - self.barrel_x, self.car_y - self.barrel_y)
        dist_pedestrian = math.hypot(self.car_x - self.pedestrian_x, self.car_y - self.pedestrian_y)

        # --- 终极奖励塑形 ---
        reward += delta_x * 50.0  
        reward -= abs(self.car_y) * 0.5
        reward -= abs(steering) * 0.2
        
        self.episode_reward += reward

        # --- 致命判定 (引入长方形车身体积考量，防穿模) ---
        if self.car_x > 4.8:
            print(f"🏆 【捷报】到达终点！总得分: {self.episode_reward:.2f}")
            reward += 100.0
            done = True
        elif abs(self.car_y) > 0.6:
            print(f"💀 【悲报】压线坠崖 (Y={self.car_y:.2f}) | 总得分: {self.episode_reward:.2f}")
            reward -= 100.0
            done = True
        elif dist_barrel < 0.30:  
            print(f"💥 【悲报】撞击静态铁桶！(距离: {dist_barrel:.2f}) | 总得分: {self.episode_reward:.2f}")
            reward -= 100.0
            done = True
        elif dist_pedestrian < 0.35: 
            print(f"🩸 【悲报】撞击动态行人！(距离: {dist_pedestrian:.2f}) | 总得分: {self.episode_reward:.2f}")
            reward -= 100.0
            done = True
            
        if self.step_count >= self.max_steps and not done:
            print(f"⏳ 【提示】20秒超时，强制重置。| 总得分: {self.episode_reward:.2f}")
            truncated = True
            
        obs = self.current_image
        info = {}
        
        return obs, reward, done, truncated, info

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()