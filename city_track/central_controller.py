#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

# 导入你的底层算法模块 (请确保它们在同级目录下或 PYTHONPATH 中)
from a_star_nav import AStarNavigator
from sac_model_nav import SACNavigator
from stabilizer_nav import Stabilizer

from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion

# --- 全局地图配置 (直接复用我们上一轮生成的矩阵) ---
CITY_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # row 0  (Y=0, 最底部的外围墙)
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 1
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 2
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 3  (S 点所在)
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 4
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 5
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 6
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # row 7  (中间横向大马路)
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 8
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 9
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 10
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 11 (E 点所在)
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # row 12
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # row 13 (Y=最大值，最顶部的外围墙)
]

class CentralController(Node):
    def __init__(self):
        super().__init__('central_controller')

        # --- 1. 状态机定义 ---
        self.STATE_ASTAR = 0
        self.STATE_SAC = 1
        self.STATE_STABILIZER = 2
        self.STATE_DONE = 3
        self.STATE_CRASHED = 4  # 👑 新增：碰撞报废状态
        
        self.current_state = self.STATE_ASTAR

        # --- 2. 传感器与执行器设置 ---
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 任务信息
        self.start_pose = (9.6, 6.6)  # 物理坐标起点 (网格 row 10, col 4)
        self.goal_pose = (31.2, 25.8) # 物理坐标终点 (网格 row 2, col 13)
        self.goal_tolerance = 0.5     # 到达终点的判定半径 (米)
        
        # 👑 【新增】：创建上帝之手服务客户端
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /gazebo/set_entity_state 服务上线...')
            
        # 👑 【新增】：在一切算法开始前，先把车挪回起点
        self.reset_car_to_start()

        # 主控制循环 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        # --- 3. 车辆位姿与状态变量 ---
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.obstacle_detected = False
        self.latest_laser_msg = None
        self.log_counter = 0
        

        # --- 4. 实例化底层算法大脑 (无ROS通信开销) ---
        self.astar_nav = AStarNavigator(grid_map=CITY_MAP, resolution=2.4)
        self.sac_nav = SACNavigator(model_path="sac_bev_finetune_500000_steps.zip")
        self.stabilizer = Stabilizer()

        # 启动时进行一次全局 A* 路径规划
        self.get_logger().info("🧠 正在进行全局 A* 路径规划...")
        self.astar_nav.plan_path(self.start_pose, self.goal_pose)
        self.get_logger().info("✅ 中央控制器初始化完毕，小车准备出发！")

    def euler_from_quaternion(self, q):
        """四元数转欧拉角 (偏航角)"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def euler_to_quaternion(self, yaw):
        """欧拉角转四元数 (供上帝之手传送使用)"""
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

    def reset_car_to_start(self):
        """【上帝之手】将小车强制传送回起点，并摆正车头"""
        self.get_logger().info("🔄 正在将小车重置回起点 (X=9.6, Y=6.6, Yaw=90°)...")
        req = SetEntityState.Request()
        
        # 🚨 这里的名字必须和你空降小车 (spawn_entity) 时的 -entity 名字完全一致！
        # 如果你用的是官方 launch，可能是 'turtlebot3_waffle_pi'；
        # 如果用的是我之前给你写的底层命令，则是 'waffle_pi'。
        req.state.name = 'waffle_pi' 
        
        req.state.pose = Pose(
            position=Point(x=self.start_pose[0], y=self.start_pose[1], z=0.05),
            orientation=self.euler_to_quaternion(1.5708)  # 1.5708 弧度即正北方向
        )
        
        # 异步调用，防止阻塞主节点的启动
        self.set_state_client.call_async(req)
        self.get_logger().info("✅ 小车已成功空降起跑线！")

    def odom_callback(self, msg):
        """实时更新小车坐标与偏航角"""
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        self.car_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        self.latest_laser_msg = msg
        
        # --- 1. 致命物理碰撞检测 (带永久捉鬼探针与 0.22 米物理免疫底线) ---
        collision_sector_indices = list(range(0, 45)) + list(range(315, 360))
        
        min_r = 10.0
        min_angle = -1
        
        # 遍历危险扇区，找出距离最近的那个点的具体角度
        for i in collision_sector_indices:
            r = msg.ranges[i]
            # 0.22米内（自身车壳）坚决无视！只记录 0.22 米以外的最危险点
            if 0.22 < r < 10.0:
                if r < min_r:
                    min_r = r
                    min_angle = i
        
        # 真正的死亡红线推至 0.25 米
        if min_r < 0.25:
            if self.current_state != self.STATE_CRASHED:
                self.get_logger().error(f"💥 发生严重碰撞！距外部障碍仅: {min_r:.2f} 米！强制锁定！")
                
                # 👑 永久保留的抓鬼探针
                self.get_logger().error(f"🔍 [抓鬼探针] 致命碰撞点位于小车的 {min_angle} 度方向！")
                self.get_logger().error(f"🔍 [抓鬼探针] 此时车身坐标: X={self.car_x:.2f}, Y={self.car_y:.2f}, Yaw={math.degrees(self.car_yaw):.1f}°")
                
                self.current_state = self.STATE_CRASHED
                self.publish_velocity(0.0, 0.0)
            return

        # --- 2. 提取并聚类全景前向雷达点云 (视野彻底放开到 3.0 米) ---
        points = []
        # 核心逻辑：只看车头前方 180 度 [-90°, 90°]。只要物体出了这个范围，就说明它在车后方！
        for i in range(270, 360):
            r = msg.ranges[i]
            if 0.22 < r <= 3.0:  # 极限视距拉满到 3.0 米
                angle = math.radians(i)
                points.append((r * math.cos(angle), r * math.sin(angle), r))
        for i in range(0, 91):
            r = msg.ranges[i]
            if 0.22 < r <= 3.0:
                angle = math.radians(i)
                points.append((r * math.cos(angle), r * math.sin(angle), r))
        
        clusters = []
        if points:
            current_cluster = [points[0]]
            for i in range(1, len(points)):
                prev_x, prev_y, _ = points[i-1]
                curr_x, curr_y, _ = points[i]
                if math.hypot(curr_x - prev_x, curr_y - prev_y) < 0.6:
                    current_cluster.append(points[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [points[i]]
            clusters.append(current_cluster)
        
        # --- 3. 极简仲裁：是否有障碍物在前方？ ---
        obstacle_in_front = False
        
        for cluster in clusters:
            min_r = min(p[2] for p in cluster)
            
            # 动态密度过滤：3米外的激光极其发散，2个点就得认；近处需要多一点防噪
            if min_r >= 1.5 and len(cluster) < 2:
                continue
            if 0.7 <= min_r < 1.5 and len(cluster) < 3:
                continue
            if min_r < 0.7 and len(cluster) < 5:
                continue
                
            cluster_width = math.hypot(cluster[0][0] - cluster[-1][0], cluster[0][1] - cluster[-1][1])
            
            # 识别为真实局部障碍物 (宽度小于 0.8 米，无视宽大的墙壁)
            if cluster_width < 0.8:
                obstacle_in_front = True
                break  # 只要前方视野内发现一个，就立刻拉响警报

        # --- 4. 状态机流转 (允许抽搐，过线即交接) ---
        if obstacle_in_front:
            if not self.obstacle_detected:
                self.get_logger().warn("🚨 在 3.0 米视野内发现前方障碍！移交 SAC！")
                self.obstacle_detected = True
        else:
            # 只要 obstacle_in_front 是 False，说明前方 180 度干干净净
            # 障碍物必然是落到了车身后方 (角度 > 90 且 < 270)
            if self.obstacle_detected:
                self.get_logger().info("✅ 障碍物已落后于车身或离开视野，立即移交拉直。")
                self.obstacle_detected = False

    def check_goal_reached(self):
        """判断是否到达终点"""
        dist_to_goal = math.hypot(self.goal_pose[0] - self.car_x, self.goal_pose[1] - self.car_y)
        if dist_to_goal < self.goal_tolerance:
            return True
        return False

    def publish_velocity(self, v, w):
        """统一的速度发布接口"""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def control_loop(self):
        """
        主控制循环 (状态机仲裁中心)
        极其严格的时序控制，决定当前谁来接管小车
        """

        # 👑 0. 撞车或任务结束，直接死刹车，不再执行任何控制逻辑
        if self.current_state == self.STATE_CRASHED or self.current_state == self.STATE_DONE:
            self.publish_velocity(0.0, 0.0)
            return

        # 👑 【新增日志探针】：每 10 帧 (0.5秒) 打印一次核心状态
        self.log_counter += 1
        if self.log_counter % 10 == 0:
            state_dict = {
                self.STATE_ASTAR: "A* 巡航 (A-Star)",
                self.STATE_SAC: "SAC 避障 (RL)",
                self.STATE_STABILIZER: "稳定器拉直 (Stabilizer)",
                self.STATE_DONE: "任务完成 (Done)"
            }
            current_state_str = state_dict.get(self.current_state, "未知状态")
            
            # 获取 A* 当前正在追逐的目标点索引和坐标
            target_info = "无"
            if hasattr(self, 'astar_nav') and self.astar_nav.current_target_index < len(self.astar_nav.current_path):
                tx, ty = self.astar_nav.current_path[self.astar_nav.current_target_index]
                target_info = f"索引 {self.astar_nav.current_target_index} -> ({tx:.2f}, {ty:.2f})"

            self.get_logger().info(
                f"📊 [监控] 状态: {current_state_str} | "
                f"车坐标: ({self.car_x:.2f}, {self.car_y:.2f}) | "
                f"车头朝向: {math.degrees(self.car_yaw):.1f}° | "
                f"A*追逐目标: {target_info}"
            )

        # 👑 【核心修复】：无论当前是谁在开车，强制 SAC 在后台“看路”，保持 4 帧记忆鲜活！
        if self.latest_laser_msg is not None:
            self.sac_nav.update_frame_stack(self.latest_laser_msg)

        # 0. 终点判定最高优先级
        if self.current_state != self.STATE_DONE and self.check_goal_reached():
            self.get_logger().info("🏆 到达终点！任务圆满结束。")
            self.current_state = self.STATE_DONE
            self.publish_velocity(0.0, 0.0)
            return

        if self.current_state == self.STATE_DONE:
            self.publish_velocity(0.0, 0.0)
            return

        # 1. 状态机流转逻辑
        if self.current_state == self.STATE_ASTAR:
            if self.obstacle_detected:
                self.get_logger().error("⚡ 剥夺 A* 控制权，SAC 紧急接管！")
                self.current_state = self.STATE_SAC
            else:
                # 正常 A* 巡航
                v, w = self.astar_nav.compute_velocity(self.car_x, self.car_y, self.car_yaw)
                self.publish_velocity(v, w)

        elif self.current_state == self.STATE_SAC:
            if not self.obstacle_detected:
                self.get_logger().info("✅ 避障完成，移交稳定器拉直车身...")
                self.current_state = self.STATE_STABILIZER
            else:
                # 👑 修改这里的调用，直接取动作，不传参数
                v, w = self.sac_nav.compute_velocity()
                self.publish_velocity(v, w)

        elif self.current_state == self.STATE_STABILIZER:
            self.astar_nav.update_nearest_target(self.car_x, self.car_y)
            # 获取 A* 的下一个目标点
            if self.astar_nav.current_target_index < len(self.astar_nav.current_path):
                target_x, target_y = self.astar_nav.current_path[self.astar_nav.current_target_index]
                is_stable, v, w = self.stabilizer.align_vehicle(self.car_x, self.car_y, self.car_yaw, target_x, target_y)
                self.publish_velocity(v, w)
                
                if is_stable:
                    self.get_logger().info("🛣️ 车身已摆正，交还 A* 巡航。")
                    self.current_state = self.STATE_ASTAR
            else:
                # 如果没目标点了，直接切回去
                self.current_state = self.STATE_ASTAR


def main(args=None):
    rclpy.init(args=args)
    node = CentralController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("程序被用户强行终止。")
    finally:
        node.publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()