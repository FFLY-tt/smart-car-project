import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
import math
import time

class PedestrianMover(Node):
    def __init__(self):
        super().__init__('pedestrian_mover')
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 Gazebo 通信服务启动...')
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.start_time = time.time()

    def timer_callback(self):
        current_time = time.time() - self.start_time
        
        # 【核心修改 1】：振幅设为 0.4 (刚好在路边缘)，角频率 0.3 (最高移动速度约为 0.12 m/s)
        y_position = 0.4 * math.cos(0.9 * current_time)
        request = SetEntityState.Request()
        request.state.name = 'dynamic_pedestrian'
        request.state.pose.position.x = 4.0  
        request.state.pose.position.y = y_position
        
        # 【核心修改 2】：必须修复 Z 坐标，让 0.175 米高的小人双脚沾地！
        request.state.pose.position.z = 0.0875
        
        # 保持行人直立的四元数 (不加的话有些版本的 Gazebo 会让它倒下)
        request.state.pose.orientation.x = 0.0
        request.state.pose.orientation.y = 0.0
        request.state.pose.orientation.z = 0.0
        request.state.pose.orientation.w = 1.0
        
        self.future = self.client.call_async(request)

def main():
    rclpy.init()
    node = PedestrianMover()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()