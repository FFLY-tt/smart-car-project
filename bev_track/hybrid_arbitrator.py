import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import cv2
from collections import deque
from stable_baselines3 import SAC

# 引入你的稳定器逻辑
from core.base_controller import BaseController

class HybridArbitrator(Node):
    def __init__(self):
        super().__init__('hybrid_arbitrator_node')
        
        # --- 1. 核心发布与订阅 ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # --- 2. 状态机变量 ---
        # 0: A* 巡航 | 1: RL 避障 | 2: 稳定恢复
        self.current_state = 0 
        self.car_x, self.car_y, self.car_yaw = 0.0, 0.0, 0.0
        self.obstacle_detected = False
        
        # --- 3. RL 大脑与数据管道 ---
        # 留空：等你 50 万步模型跑完，填入模型路径
        self.rl_model_path = "sac_bev_rebuild_final.zip" 
        self.rl_model = None # 稍后加载
        # 手动实现 VecFrameStack(n_stack=4) 的逻辑
        self.frame_stack = deque(maxlen=4)
        for _ in range(4):
            self.frame_stack.append(np.zeros((64, 64, 1), dtype=np.uint8))
            
        # --- 4. A* 路径与稳定器 ---
        # 队友 A* 算法的输出接口（暂时 Mock 几个点用于测试）
        self.waypoints = [(1.0, 0.0), (2.0, 0.0), (3.0, 1.0), (4.0, 1.0)]
        self.current_target_index = 0
        self.base_controller = BaseController(k_y=0.5, k_yaw=1.0)
        
        # 50Hz 控制循环
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("【系统】混合动力仲裁器已启动。等待接入 RL 模型...")

    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.car_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def scan_callback(self, msg):
        # 1. 极其精准的“警戒区”雷达检测 (前方 +- 30度，1.2米以内)
        danger_zone = False
        ranges = msg.ranges
        for i, dist in enumerate(ranges):
            if math.isinf(dist) or math.isnan(dist) or dist < 0.15:
                continue
            
            # 将索引转换为角度 (弧度)
            angle = msg.angle_min + i * msg.angle_increment
            # 如果在车头正前方 +- 30度 (约 0.52 弧度) 且距离小于 1.2米
            if abs(angle) < 0.52 and dist < 1.2:
                danger_zone = True
                break # 发现一个障碍物就足够触发了
        
        self.obstacle_detected = danger_zone

        # 2. 同时生成 BEV 图像（给 RL 模型备用）
        bev_image = np.zeros((64, 64, 1), dtype=np.uint8)
        center_x, center_y = 32, 32
        pixels_per_meter = 32 / 3.0
        for i, dist in enumerate(ranges):
            if math.isinf(dist) or math.isnan(dist) or dist > 3.0 or dist < 0.15:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            img_y = int(center_y - (dist * math.cos(angle)) * pixels_per_meter) 
            img_x = int(center_x - (dist * math.sin(angle)) * pixels_per_meter) 
            if 0 <= img_x < 64 and 0 <= img_y < 64:
                bev_image[img_y, img_x, 0] = 255
        
        # 压入双端队列，保持最新的 4 帧
        self.frame_stack.append(bev_image)

    def control_loop(self):
        twist = Twist()

        # --- 状态转移逻辑 ---
        if self.obstacle_detected:
            if self.current_state != 1:
                self.get_logger().warn("🚨 障碍物侵入！剥夺 A* 控制权，切换至 State 1 (RL 接管)")
            self.current_state = 1
        else:
            if self.current_state == 1:
                self.get_logger().info("✅ 障碍物解除。切换至 State 2 (稳定器介入)")
                self.current_state = 2

        # --- 执行状态对应动作 ---
        if self.current_state == 0:
            twist = self.run_a_star_tracker()
            
        elif self.current_state == 1:
            twist = self.run_rl_avoidance()
            
        elif self.current_state == 2:
            twist, stabilized = self.run_stabilizer()
            if stabilized:
                self.get_logger().info("🛣️ 姿态已恢复。切换回 State 0 (A* 巡航)")
                self.current_state = 0

        self.cmd_pub.publish(twist)

    def run_a_star_tracker(self):
        """队友需要完善的 A* 循迹控制器"""
        twist = Twist()
        # 这里写一个极其简单的循迹作为占位符
        if self.current_target_index < len(self.waypoints):
            target_x, target_y = self.waypoints[self.current_target_index]
            dist = math.hypot(target_x - self.car_x, target_y - self.car_y)
            
            if dist < 0.2: # 到达当前点，切换下一个点
                self.current_target_index += 1
            else:
                # 计算目标角度并简单 P 控制
                target_angle = math.atan2(target_y - self.car_y, target_x - self.car_x)
                angle_error = target_angle - self.car_yaw
                # 将角度误差规范化到 [-pi, pi]
                angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
                
                twist.linear.x = 0.2
                twist.angular.z = float(1.5 * angle_error)
        else:
            # 跑完全程，停车
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        return twist

    def run_rl_avoidance(self):
        """挂载 RL 大脑"""
        twist = Twist()
        if self.rl_model is None:
            # 避免模型还没训练好时报错
            twist.linear.x = 0.0 
            return twist
            
        # 将 4 个单通道图像堆叠成 1 个 4通道图像 (64, 64, 4)
        stacked_obs = np.concatenate(list(self.frame_stack), axis=-1)
        
        # 送入模型推理 (deterministic=True 极其关键)
        action, _ = self.rl_model.predict(stacked_obs, deterministic=True)
        
        # 必须使用与训练时完全一致的物理逆变换
        twist.linear.x = float((action[0] + 1.0) / 2.0 * 0.25)
        twist.angular.z = float(action[1] * 1.0)
        return twist

    def run_stabilizer(self):
        """稳定恢复：修正横向和航向误差"""
        twist = Twist()
        stabilized = False
        
        if self.current_target_index >= len(self.waypoints):
            return twist, True

        # 拿到 A* 的下一个目标点
        target_x, target_y = self.waypoints[self.current_target_index]
        
        # 计算小车当前与目标点的误差
        target_angle = math.atan2(target_y - self.car_y, target_x - self.car_x)
        yaw_error = target_angle - self.car_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error)) # 规范化
        
        # 这里简化：用目标点的 Y 作为参考线
        y_error = self.car_y - target_y 
        
        # 如果误差足够小，宣告稳定完成
        if abs(y_error) < 0.1 and abs(yaw_error) < 0.1:
            stabilized = True
        else:
            # 调用你写的 base_controller 的核心算法
            twist.linear.x = 0.1 # 恢复期间低速行驶
            twist.angular.z = float(-self.base_controller.k_y * y_error - self.base_controller.k_yaw * yaw_error)
            
        return twist, stabilized

def main(args=None):
    rclpy.init(args=args)
    node = HybridArbitrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()