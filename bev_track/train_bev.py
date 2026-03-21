import rclpy
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_track.auto_car_env_bev import BEVCarEnv
from bev_track.gap_sac_model import GAPExtractor

def main():
    print(f"【系统自检】PyTorch CUDA: {torch.cuda.is_available()}")
    print("【系统】正在初始化 BEV + GAP-SAC 终极架构...")
    
    env = BEVCarEnv()
    check_env(env)
    env = Monitor(env)
    
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 将你的 GAP 模块作为自定义网络插入
    policy_kwargs = dict(
        features_extractor_class=GAPExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    print("【系统】正在唤醒带有门控注意力的 SAC 大脑...")
    model = SAC("CnnPolicy", 
                    env,
                    policy_kwargs=policy_kwargs, 
                    learning_rate=3e-4,  
                    buffer_size=100000,   # 池子加大，存更多好数据
                    learning_starts=500,
                    batch_size=256,       # 【提速核弹】：每次抽取 512 条经验同时塞进 GPU 训练 (默认是256)
                    train_freq=4,         # 每走 1 步就训练一次
                    gradient_steps=1,     # 【贪婪模式】：每走 1 步，强行更新 2 次神经网络权重！加速榨取经验！
                    verbose=1, 
                    tensorboard_log="./logs/tensorboard/")

    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./logs/', name_prefix='sac_bev_model')

    print("🔥 【系统】BEV 架构点火成功！开始极速训练...")
    try:
        # 明确打上标签，方便在 TensorBoard 里对比碾压纯视觉
        model.learn(total_timesteps=150000, callback=checkpoint_callback, log_interval=1, tb_log_name="BEV_GAP_SAC")
    except KeyboardInterrupt:
        print("\n【系统】手动中断...")
    finally:
        model.save("sac_bev_final")
        env.close()

if __name__ == '__main__':
    main()