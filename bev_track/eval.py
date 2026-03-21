import rclpy
from stable_baselines3 import SAC
# 👑 【核心补丁 1】：引入必要的维度转换器
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
import time
import os
import sys

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_track.auto_car_env_bev import BEVCarEnv
from bev_track.gap_sac_model import GAPExtractor

def main():
    print("【系统】正在初始化 BEV 评估环境...")
    env = BEVCarEnv()
    
    # 👑 【核心补丁 2】：给测试环境套上和训练时一模一样的“4帧视觉暂留”外挂
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # 加载初代极品脑电波
    model_path = "./logs/checkpoints/sac_bev_rebuild_140000_steps.zip"
    
    if not os.path.exists(model_path):
        print(f"🚨 找不到模型文件 {model_path}！请确认路径。")
        return

    print(f"🧠 【系统】正在加载极品脑电波: {model_path}")
    model = SAC.load(model_path, env=env)
    
    print("🔥 【系统】阅兵开始！")
    
    # 连续测试 10 个回合
    for episode in range(10):
        # 👑 【核心补丁 3】：VecEnv 的 reset 只返回 obs，没有 info
        obs = env.reset()
        done = [False]  # VecEnv 的 done 是一个数组
        total_reward = 0
        step_count = 0
        
        while not done[0]:
            # deterministic=True 关闭随机探索，展现完美走线
            action, _states = model.predict(obs, deterministic=True) 
            
            # VecEnv 的 step 返回 4 个值，且都是数组形式
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
        print(f"🏁 回合 {episode + 1}/10 结束 | 存活步数: {step_count} | 总得分: {total_reward:.2f}")
        time.sleep(1.0) # 回合间歇

    env.close()

if __name__ == '__main__':
    main()