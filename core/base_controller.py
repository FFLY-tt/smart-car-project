import numpy as np

class BaseController:
    def __init__(self, k_y=2.0, k_yaw=1.5):
        """
        双闭环基础控制器
        :param k_y: 距离误差纠正系数
        :param k_yaw: 角度误差纠正系数
        """
        self.k_y = k_y
        self.k_yaw = k_yaw

    def get_base_action(self, car_y, car_yaw, dist_barrel, dist_pedestrian):
        # --- 1. 双闭环寻迹先验 ---
        # 目标 Y 是 0，目标 Yaw 是 0。
        # 同时纠正偏移距离和车头朝向，这才能让车“死死咬住”中线
        theta_base = -self.k_y * car_y - self.k_yaw * car_yaw
        
        theta_base = float(np.clip(theta_base, -1.2, 1.2))

        # --- 2. 速度先验 (自适应巡航) ---
        min_dist = min(dist_barrel, dist_pedestrian)

        if min_dist > 3.0:
            v_base = 0.4  
        elif min_dist > 1.2:
            v_base = 0.2  
        else:
            v_base = 0.05 

        return v_base, theta_base