import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
# 导入你刚刚辛辛苦苦写好的环境！
import sys
import os
# 将项目根目录临时加入系统路径，确保能跨文件夹导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_track.auto_car_env_vision import AutonomousCarEnv
def main():
    print(f"【系统硬件自检】当前 PyTorch 是否使用外星人 GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"【系统硬件自检】识别到的显卡型号: {torch.cuda.get_device_name(0)}")
        
    print("【系统】正在初始化物理世界与 AI 大脑的连接桥梁...")
    
    # 1. 实例化你的定制环境
    env = AutonomousCarEnv()

    # 2.环境健康检查
    # SB3 会用极其严苛的标准，检查你的 reset() 和 step() 输出的格式对不对
    print("【系统】正在对环境进行 Gymnasium 标准化质检...")
    check_env(env)
    print("【系统】质检通过！环境接口完美兼容。")

    env = Monitor(env)  # 封装之前加上监控
    # 3. 向量化包装 (Stable Baselines3 的标准要求)
    env = DummyVecEnv([lambda: env])

    # 4. 【核心注入：短时记忆】
    # 将过去的 4 帧画面堆叠在一起。
    # 物理意义：AI 不再只看一张图，而是看一部“微型动画”。
    # 它能通过前 3 帧铁桶的残影，记住盲区里到底有没有东西！
    env = VecFrameStack(env, n_stack=4)

    # 5. 通道转换 (适配 PyTorch)
    # 注意：这一步必须放在 FrameStack 之后！
    # 它会将堆叠后的 (128, 128, 12通道) 转换为 PyTorch 需要的 (12通道, 128, 128)
    env = VecTransposeImage(env)



    # 3. 实例化 SAC 算法大脑
    # "CnnPolicy" 告诉大脑：你的眼睛看到的是图片，请自动启用卷积神经网络。
    # verbose=1 会在终端里打印训练的详细损失率和得分。
    # buffer_size 是经验回放池，因为我们只做轻量级测试，先设为 10000。
    # 将 buffer_size 从 10000 提升至 50000 (如果服务器内存足够大，甚至可以设为 100000)
    # 增加 optimize_memory_usage=True 以防止服务器内存爆满
    print("【系统】正在唤醒 SAC 深度强化学习神经网络...")
    model = SAC("CnnPolicy", 
                env,
                learning_rate=1e-4,  # 手动强制注入极其稳健的学习率
                verbose=1, 
                buffer_size=50000,
                learning_starts=100,  # 先随机乱开 100 步收集一点初始数据
                tensorboard_log="./logs/tensorboard/vision_run/")  # 开启可视化日志

    # 4. 设置自动保存机制
    # 每训练 2000 步，自动保存一次脑电波 (权重模型)，防止突然断电白跑
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./logs/', name_prefix='sac_car_model')

    # 5. 点火！开始疯狂试错与训练！
    # total_timesteps = 50000 意味着小车要在物理世界里做 5 万次决策
    print("🔥 【系统】点火成功！开始自动化训练循环...")
    try:
        model.learn(total_timesteps=500000,
                    callback=checkpoint_callback,
                    log_interval=1,
                    tb_log_name="vision_baseline")
    except KeyboardInterrupt:
        print("\n【系统】接收到手动中断信号，正在提前结束训练...")
    finally:
        # 6. 训练结束，保存最终的大脑切片
        print("💾 【系统】正在保存最终模型到 sac_autonomous_car.zip...")
        model.save("sac_autonomous_car")
        
        # 优雅关闭环境
        env.close()
        print("【系统】训练程序安全退出。")

if __name__ == '__main__':
    main()