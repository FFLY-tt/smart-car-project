import rclpy
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_track.auto_car_env_bev import BEVCarEnv
from bev_track.gap_sac_model import GAPExtractor

def main():
    env = BEVCarEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    policy_kwargs = dict(
        features_extractor_class=GAPExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='./logs/checkpoints/', 
        name_prefix='sac_bev_finetune'
    )

    model = SAC("CnnPolicy", 
                env,
                policy_kwargs=policy_kwargs, 
                learning_rate=3e-4,  
                buffer_size=50000,   
                learning_starts=500,
                batch_size=256,       
                train_freq=4,         
                gradient_steps=1,     
                verbose=1, 
                tensorboard_log="./logs/tensorboard/")
    
    try:
        model.learn(total_timesteps=500000, callback=checkpoint_callback, log_interval=1, tb_log_name="BEV_GAP_SAC_Rebuild")
    except KeyboardInterrupt:
        pass
    finally:
        model.save("sac_bev_rebuild_final")
        env.close()

if __name__ == '__main__':
    main()