#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion
import math

class PedestrianController(Node):
    def __init__(self):
        super().__init__('pedestrian_controller')
        
        # 1. 创建 Service Client (而不是 Publisher)
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # 等待服务上线
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /gazebo/set_entity_state 服务上线...')
            
        # 2. 控制参数
        self.update_rate_hz = 30.0  # 30Hz 对于服务调用来说非常安全且平滑
        self.speed_mps = 0.15        # 行人匀速行走速度

        self.p1_params = {
            'name': 'P1_pedestrian',
            'fixed_y': 12.0,
            'fixed_z': 0.0875,
            'start_x': 8.6,
            'end_x': 10.6,
            'current_x': 8.6,
            'direction': 1,
            'yaw': 0.0
        }

        self.p2_params = {
            'name': 'P2_pedestrian',
            'fixed_x': 24.0,
            'fixed_z': 0.0875,
            'start_y': 15.8,
            'end_y': 17.8,
            'current_y': 15.8,
            'direction': 1,
            'yaw': 1.5708
        }

        self.timer = self.create_timer(1.0 / self.update_rate_hz, self.timer_callback)
        self.last_time = self.get_clock().now()
        
        self.get_logger().info('行人控制节点已启动，行人开始通过服务调用移动！')

    def euler_to_quaternion(self, yaw):
        """欧拉角转四元数"""
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

    def send_state_request(self, params, current_x, current_y):
        """发送异步服务请求"""
        req = SetEntityState.Request()
        req.state.name = params['name']
        req.state.pose = Pose(
            position=Point(x=current_x, y=current_y, z=params['fixed_z']),
            orientation=self.euler_to_quaternion(params['yaw'])
        )
        # 使用异步调用，防止阻塞主线程 (ROS 2 最佳实践)
        self.client.call_async(req)

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        dist_to_move = self.speed_mps * dt

        # --- A. 更新 P1 状态 (X轴移动) ---
        p1 = self.p1_params
        p1['current_x'] += dist_to_move * p1['direction']
        if p1['current_x'] >= p1['end_x']:
            p1['current_x'] = p1['end_x']
            p1['direction'] = -1
        elif p1['current_x'] <= p1['start_x']:
            p1['current_x'] = p1['start_x']
            p1['direction'] = 1
        
        self.send_state_request(p1, p1['current_x'], p1['fixed_y'])

        # --- B. 更新 P2 状态 (Y轴移动) ---
        p2 = self.p2_params
        p2['current_y'] += dist_to_move * p2['direction']
        if p2['current_y'] >= p2['end_y']:
            p2['current_y'] = p2['end_y']
            p2['direction'] = -1
        elif p2['current_y'] <= p2['start_y']:
            p2['current_y'] = p2['start_y']
            p2['direction'] = 1
            
        self.send_state_request(p2, p2['fixed_x'], p2['current_y'])

def main(args=None):
    rclpy.init(args=args)
    controller = PedestrianController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()