# smart-car Project Documentation (Phase 1)


Discription: 
基于残差强化学习的端到端自动驾驶仿真系统
阶段1：只进行局部避障，动态行人躲避和静态障碍物躲避，基于简单的直线公路场景

End-to-end autonomous driving simulation system based on residual reinforcement learning
Phase 1:
Phase 1: Local obstacle avoidance is implemented, including dynamic pedestrian avoidance and static obstacle avoidance, based on a simple straight-line road scenario.

## 1. 核心目标 (Core Objective)
本项目旨在 ROS 2 与 Gazebo 构建的高保真物理仿真环境中，开发一套高效的自动驾驶控制系统。系统要求车辆在一条包含静态障碍物（路障）和动态障碍物（横穿行人）的赛道上，实现稳定的车道保持与自主避障。
This project aims to develop an efficient autonomous driving control system within a high-fidelity physics simulation environment built using ROS 2 and Gazebo. The system requires the vehicle to achieve stable lane keeping and autonomous obstacle avoidance on a track containing both static obstacles (roadblocks) and dynamic obstacles (pedestrians crossing).

## 2. 核心架构：残差控制融合 (Residual Control Architecture)
为了解决纯端到端深度学习极易产生的“轨迹摇摆”和“收敛困难”问题，本项目摒弃了传统的单一大脑控制，采用工业级自动驾驶的混合控制架构：

基础物理控制器 (Base Controller / 兜底机制)： 不依赖视觉输入，仅依靠底层里程计 (Odometry) 数据。通过双闭环 PID 算法 (位置偏差 + 偏航角纠正) 实现绝对平滑的直线寻迹，并通过简易的自适应巡航 (ACC) 逻辑控制基础限速。

深度强化学习网络 (RL Agent / 视觉避障决策)： 采用 SAC (Soft Actor-Critic) 算法搭配 CNN 策略。输入为 128x128 的三通道 RGB 高清画面，输出为对基础动作的**“残差微调量 (Residual Actions)”**。神经网络无需从零学习物理学直线行驶，仅专注于理解图像语义并在遇到障碍物时输出紧急避险转向。

To address the "trajectory swaying" and "convergence difficulties" inherent in pure end-to-end deep learning, this project abandons traditional single-brain control and adopts a hybrid control architecture for industrial-grade autonomous driving:

**Base Controller (Backup Mechanism):** Relies solely on underlying odometry data, without visual input. It achieves absolutely smooth straight-line tracking through a dual-closed-loop PID algorithm (position deviation + yaw angle correction) and controls basic speed limits through simple adaptive cruise control (ACC) logic.

**Deep Reinforcement Learning Network (RL Agent /Visual Obstacle Avoidance Decision-Making):** Employs a SAC (Soft Actor-Critic) algorithm combined with a CNN strategy. Input is a 128x128 three-channel RGB high-definition image, and output is **"residual actions"**. The neural network does not need to learn the physics of straight-line driving from scratch; it focuses solely on understanding image semantics and outputting emergency avoidance steering when encountering obstacles.

## 3. 关键技术特性 (Key Technical Features)

域随机化 (Domain Randomization)： 每一回合强制随机刷新静态障碍物的坐标，彻底封死神经网络“死记硬背地图”的作弊路径，迫使模型学习真实的视觉泛化能力。

物理流速与算法推断解耦 (Time-Scaling Synchronization)： 突破物理引擎的 1.0 倍真实时间限制，实现 3 倍速仿真加速，并在此基础上重构了 Python 端的帧率休眠逻辑与动态障碍物的相对运动角频率，极大提升算力利用率。

高维度死亡判定 (Strict Collision Physics)： 摒弃质点碰撞模型，将车辆的物理长宽尺寸及高速运动补偿纳入环境裁判逻辑，消除穿模效应，提供极度严苛的奖励塑形 (Reward Shaping)。

Domain Randomization: Forces a random refresh of static obstacle coordinates each round, completely blocking the neural network's "rote memorization" cheating path and forcing the model to learn realistic visual generalization capabilities.

Time-Scaling Synchronization: Breaks through the 1.0x real-time limitation of the physics engine, achieving 3x simulation acceleration. Furthermore, it reconstructs the frame rate sleep logic and the relative motion angular frequency of dynamic obstacles on the Python side, greatly improving computational efficiency.

Strict Collision Physics: Abandons the point mass collision model, incorporating the vehicle's physical dimensions and high-speed motion compensation into the environmental judging logic, eliminating clipping effects and providing extremely rigorous reward shaping.



### version history

