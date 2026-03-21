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
    # PyTorch 需要 (Channel, H, W)，此工具会自动转换
    env = VecTransposeImage(env)

    # 👑 【核心注入】：替换 SB3 默认网络，注入你的门控注意力网络
    policy_kwargs = dict(
        features_extractor_class=GAPExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    print("【系统】正在唤醒带有门控注意力的 SAC 大脑...")
    model = SAC("CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,  # 注入！
                learning_rate=1e-4,
                verbose=1,
                buffer_size=50000,
                learning_starts=100,
                tensorboard_log="./logs/tensorboard/bev_gap_run/")  # 独立的日志目录

    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./logs/', name_prefix='gap_sac_bev')

    print("🔥 【系统】BEV 架构点火成功！开始自动化极速训练...")
    try:
        model.learn(total_timesteps=100000, callback=checkpoint_callback, log_interval=1)
    except KeyboardInterrupt:
        print("\n【系统】接收到手动中断信号...")
    finally:
        model.save("gap_sac_bev_final")
        env.close()


if __name__ == '__main__':
    main()