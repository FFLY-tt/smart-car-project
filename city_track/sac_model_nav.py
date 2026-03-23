import sys
import numpy as np
import math
from collections import deque
from stable_baselines3 import SAC

# 内存级欺骗模块：解决模型寻找旧文件夹的问题
import gap_sac_model 
from gap_sac_model import GAPExtractor
sys.modules['bev_track'] = gap_sac_model
sys.modules['bev_track.gap_sac_model'] = gap_sac_model

class SACNavigator:
    def __init__(self, model_path="sac_bev_finetune_500000_steps.zip"):
        print(f"🧠 [SAC] 正在挂载带有 GAP 注意力机制的神经网络: {model_path}...")
        try:
            self.model = SAC.load(model_path)
            print("✅ [SAC] 神经网络挂载成功！")
        except Exception as e:
            print(f"❌ [FATAL] 模型加载失败！出于物理安全考虑，系统拒绝启动。")
            print(f"🔍 真实的底层报错信息是: {e}")
            sys.exit(1)

        # 初始化 4 帧全黑队列
        self.frame_stack = deque(
            [np.zeros((64, 64, 1), dtype=np.uint8) for _ in range(4)], 
            maxlen=4
        )

    def lidar_to_bev(self, scan_ranges, angle_min, angle_increment):
        bev_image = np.zeros((64, 64, 1), dtype=np.uint8)
        center_x, center_y = 32, 32
        max_dist = 3.0
        pixels_per_meter = 32 / max_dist 
        
        for i, dist in enumerate(scan_ranges):
            if math.isinf(dist) or math.isnan(dist) or dist > max_dist or dist < 0.15:
                continue
            angle = angle_min + i * angle_increment
            img_y = int(center_y - (dist * math.cos(angle)) * pixels_per_meter) 
            img_x = int(center_x - (dist * math.sin(angle)) * pixels_per_meter) 
            
            if 0 <= img_x < 64 and 0 <= img_y < 64:
                bev_image[img_y, img_x, 0] = 255 
                
        return bev_image

    # 👑 让 SAC 在后台保持对物理世界的连续记忆
    def update_frame_stack(self, laser_msg):
        current_bev = self.lidar_to_bev(
            laser_msg.ranges, 
            laser_msg.angle_min, 
            laser_msg.angle_increment
        )
        self.frame_stack.append(current_bev)

    # 👑 无需传参，直接利用连贯记忆执行避障
    def compute_velocity(self):
        if self.model is None:
            return 0.0, 0.0

        obs = np.concatenate(list(self.frame_stack), axis=-1)
        obs = np.expand_dims(obs, axis=0) 

        action_batch, _states = self.model.predict(obs, deterministic=True)
        action = action_batch[0]

        v_final = (action[0] + 1.0) / 2.0 * 0.25  
        omega_final = float(action[1]) * 1.0      

        v_final = max(0.0, v_final)
        return float(v_final), float(omega_final)