import math
import heapq
import numpy as np

class AStarNavigator:
    def __init__(self, grid_map, resolution=2.4):
        """
        初始化 A* 导航器
        :param grid_map: 2D 数组，1 表示道路，0 表示障碍物
        :param resolution: 每个网格代表的实际物理宽度（米）
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.max_y = len(grid_map)
        self.max_x = len(grid_map[0])
        self.current_path = []  # 存储真实的物理路径点 [(x1, y1), (x2, y2), ...]
        self.current_target_index = 0
        
        # 循迹 (Path Follower) 参数
        self.max_v = 0.22      # 最大线速度
        self.max_w = 1.0       # 最大角速度
        self.lookahead_dist = 0.8 # 前视距离

    def _real_to_grid(self, x, y):
        col = int(x / self.resolution + 0.5)
        row = int(y / self.resolution + 0.5)
        return (row, col)

    def _grid_to_real(self, row, col):
        x = col * self.resolution
        y = row * self.resolution
        return (x, y)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan_path(self, start_pose, goal_pose):
        start_grid = self._real_to_grid(start_pose[0], start_pose[1])
        goal_grid = self._real_to_grid(goal_pose[0], goal_pose[1])

        if not (0 <= start_grid[0] < self.max_y and 0 <= start_grid[1] < self.max_x):
            print("❌ 起点超出地图边界！")
            return []
        if self.grid_map[goal_grid[0]][goal_grid[1]] == 0:
            print("❌ 终点位于障碍物内！")
            return []

        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        cost_so_far = {}
        came_from[start_grid] = None
        cost_so_far[start_grid] = 0

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == goal_grid:
                break

            for dx, dy in neighbors:
                next_node = (current[0] + dy, current[1] + dx)
                if 0 <= next_node[0] < self.max_y and 0 <= next_node[1] < self.max_x:
                    if self.grid_map[next_node[0]][next_node[1]] == 1: 
                        new_cost = cost_so_far[current] + 1
                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            priority = new_cost + self.heuristic(goal_grid, next_node)
                            heapq.heappush(frontier, (priority, next_node))
                            came_from[next_node] = current

        if goal_grid not in came_from:
            print("❌ A* 无法找到到达终点的路径！")
            return []

        grid_path = []
        current = goal_grid
        while current != start_grid:
            grid_path.append(current)
            current = came_from[current]
        grid_path.append(start_grid)
        grid_path.reverse()

        self.current_path = [self._grid_to_real(r, c) for r, c in grid_path]
        self.current_target_index = 0
        
        print(f"✅ A* 路径规划成功！共生成 {len(self.current_path)} 个途径点。")
        return self.current_path

    def update_nearest_target(self, current_x, current_y):
        if not self.current_path or self.current_target_index >= len(self.current_path):
            return
            
        min_dist = float('inf')
        closest_idx = self.current_target_index
        
        for i in range(self.current_target_index, len(self.current_path)):
            tx, ty = self.current_path[i]
            dist = math.hypot(tx - current_x, ty - current_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        target_idx = closest_idx
        lookahead_dist = 1.2 
        
        for i in range(closest_idx, len(self.current_path)):
            tx, ty = self.current_path[i]
            if math.hypot(tx - current_x, ty - current_y) >= lookahead_dist:
                target_idx = i
                break
        else:
            target_idx = len(self.current_path) - 1
            
        self.current_target_index = target_idx

    def compute_velocity(self, current_x, current_y, current_yaw):
        self.update_nearest_target(current_x, current_y)
        
        if not self.current_path or self.current_target_index >= len(self.current_path):
            return 0.0, 0.0

        target_x, target_y = self.current_path[self.current_target_index]
        dist_to_target = math.hypot(target_x - current_x, target_y - current_y)

        if dist_to_target < self.lookahead_dist:
            self.current_target_index += 1
            if self.current_target_index >= len(self.current_path):
                return 0.0, 0.0 
            target_x, target_y = self.current_path[self.current_target_index]

        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
        alpha = angle_to_target - current_yaw

        while alpha > math.pi: alpha -= 2.0 * math.pi
        while alpha < -math.pi: alpha += 2.0 * math.pi

        w = 1.5 * alpha 
        w = max(min(w, self.max_w), -self.max_w)

        if abs(alpha) > 0.8:
            v = 0.05
        else:
            v = self.max_v * (1.0 - abs(alpha) / math.pi)
            v = max(min(v, self.max_v), 0.0)

        return float(v), float(w)