import math

class Stabilizer:
    def __init__(self):
        """
        初始化姿态稳定器
        专门负责在 SAC 移交 A* 时，消除横向误差和航向误差
        """
        self.align_tolerance = 0.08  # 角度对齐容差 (约 5 度)
        self.max_w = 0.8             # 稳定的最大旋转角速度

    def align_vehicle(self, current_x, current_y, current_yaw, target_x, target_y):
        """
        计算如何将车头对准前方的目标点
        :return: (is_stable, v, w) 
                 is_stable: 布尔值，True表示已经拉直，可以交还A*了
        """
        # 1. 计算小车当前位置到目标点的理想航向角
        target_yaw = math.atan2(target_y - current_y, target_x - current_x)
        
        # 2. 计算航向误差
        yaw_error = target_yaw - current_yaw

        # 角度归一化到 [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2.0 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2.0 * math.pi

        # 3. 判断是否已经对齐
        if abs(yaw_error) < self.align_tolerance:
            # 已经对齐，稳定完成！
            return True, 0.0, 0.0

        # 4. 原地 P 控制器旋转 (仅旋转，不前进，确保安全)
        w = 1.5 * yaw_error
        
        # 截断限幅
        w = max(min(w, self.max_w), -self.max_w)
        
        # v 给个极小的速度或者直接给 0
        v = 0.02 

        return False, v, w