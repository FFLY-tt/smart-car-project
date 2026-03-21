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
        
        # --- 1. 初始化 ROS 2 节点 ---
        rclpy.init()
        self.node = rclpy.create_node('rl_bev_env_node')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        
        self.set_state_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('等待 Gazebo 传送服务启动...')
        
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.node.create_subscription(ModelStates, '/gazebo/model_states', self.models_callback, 10)
        # 👑 【核心】：订阅激光雷达，放弃臃肿的摄像头图像
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # --- 2. 状态变量 ---
        self.robot_name = 'turtlebot3_waffle_pi'
        self.car_x, self.car_y, self.car_yaw = 0.2, 0.0, 0.0
        self.pedestrian_x, self.pedestrian_y = 100.0, 100.0
        self.barrel_x, self.barrel_y = 2.0, 0.0
        
        # 物理状态记忆与礼让机制
        self.prev_x = 0.2
        self.prev_omega = 0.0 
        self.yield_timer = 0
        self.step_count = 0
        self.max_steps = 200      
        self.episode_reward = 0.0 
        
        # 👑 【核心降维】：当前状态变为 64x64 的单通道黑白 BEV 图像
        self.current_bev = np.zeros((64, 64, 1), dtype=np.uint8)
        
        self.base_controller = BaseController(k_y=0.5, k_yaw=1.0)        
        
        # --- 3. 强化学习空间定义 ---
        # 动作空间：[油门, 方向盘]，释放控制权
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观察空间：64x64 黑白占据栅格图
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

    # 👑 【极其漂亮的数据管道】：将 1D 雷达变成 2D BEV 上帝视角
    def scan_callback(self, msg):
        bev_image = np.zeros((64, 64, 1), dtype=np.uint8)
        center_x, center_y = 32, 32
        max_dist = 3.0
        pixels_per_meter = 32 / max_dist # 缩放比例
        
        ranges = msg.ranges
        for i, dist in enumerate(ranges):
            # 过滤无效噪点和车身盲区
            if math.isinf(dist) or math.isnan(dist) or dist > max_dist or dist < 0.15:
                continue
                
            angle = msg.angle_min + i * msg.angle_increment
            # ROS 坐标系：前方为 X，左方为 Y。转化为图像坐标系：
            img_y = int(center_y - (dist * math.cos(angle)) * pixels_per_meter) # 前方在图像上方
            img_x = int(center_x - (dist * math.sin(angle)) * pixels_per_meter) # 左方在图像左方
            
            if 0 <= img_x < 64 and 0 <= img_y < 64:
                bev_image[img_y, img_x, 0] = 255 # 点亮障碍物栅格
                    
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
        
        # 极简模式：固定障碍物或小范围随机
        random_barrel_y = random.uniform(-0.1, 0.1) 
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
        self.prev_omega = 0.0
        self.yield_timer = 0
        self.step_count = 0
        self.episode_reward = 0.0 
        
        return self.current_bev, {}

    def step(self, action):
        self.step_count += 1
        dist_barrel = math.hypot(self.car_x - self.barrel_x, self.car_y - self.barrel_y)
        dist_pedestrian = math.hypot(self.car_x - self.pedestrian_x, self.car_y - self.pedestrian_y)

        # --- 1. 2D 动作空间映射 ---
        v_final = (action[0] + 1.0) / 2.0 * 0.25  # 油门 [0, 0.25]
        omega_final = float(action[1]) * 1.0      # 转向 [-1.0, 1.0]
        
        # 计算平滑度 (Jerk)
        jerk = abs(omega_final - self.prev_omega)
        self.prev_omega = omega_final

        twist = Twist()
        twist.linear.x, twist.angular.z = v_final, omega_final
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.033)
        
        # --- 2. 状态结算与底层经济学逻辑 ---
        done = False
        truncated = False
        reward = 0.0
        
        delta_x = self.car_x - self.prev_x
        self.prev_x = self.car_x

        base_income = delta_x * 50.0
        
        # 极其严厉的惩罚：防压线、防乱打方向、防鬼畜摇头
        discount = (abs(self.car_y) * 1.0) + (abs(omega_final) * 0.5) + (jerk * 2.0)
        step_reward = max(0.0, base_income - discount)
        
        # 👑 礼让法则与防老赖时间锁
        if dist_pedestrian < 1.0:
            if v_final < 0.05:
                self.yield_timer += 1
                if self.yield_timer <= 30: # 只发前30步的补贴
                    step_reward += 1.0 
            elif v_final > 0.15:
                step_reward -= 2.0 # 不减速冲卡重罚
        else:
            self.yield_timer = 0 # 行人离开，重置计时器

        reward += (step_reward + 0.01)
        self.episode_reward += reward

        # 致命红线
        if self.car_x > 4.8:
            print(f"🏆 【捷报】到达终点！总得分: {self.episode_reward:.2f}")
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