#!/bin/bash

# 获取当前工作目录的绝对路径，确保后续 Python 脚本能找到正确的位置
PROJECT_ROOT=$(pwd)

echo "🚀 【系统】正在一键点火，启动端到端自动驾驶仿真环境..."

# -------------------------------------------------------------------------
# Tab 1: 启动 Gazebo 物理世界
# 注意：把这里的命令替换为你平时启动那个包含 5 米赛道 world 文件的命令
# 例如: ros2 launch gazebo_ros gazebo.launch.py world:=/你的路径/custom_highway.world
# -------------------------------------------------------------------------
gnome-terminal --tab --title="🌍 Gazebo World" -- bash -c "
source /opt/ros/humble/setup.bash;
echo '【阶段 1】正在加载物理引擎与 1:10 微缩赛道...';

ros2 launch gazebo_ros gazebo.launch.py world:=${PROJECT_ROOT}/worlds/custom_highway.world;

exec bash"

# 极其关键的延时：必须给 Gazebo 物理引擎留出 5 秒钟的初始化时间，否则小车生成会崩溃
echo "⏳ 等待物理引擎稳定 (5秒)..."
sleep 5

# -------------------------------------------------------------------------
# Tab 2: 降临红色小车 (破解幽灵缓存并指定精确坐标)
# -------------------------------------------------------------------------
gnome-terminal --tab --title="🚗 Spawn Car" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle_pi;
echo '【阶段 2】正在将车辆投放至起跑线 (X=0.2)...';
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py x_pose:=0.2 y_pose:=0.0 z_pose:=0.05;
exec bash"

# 等待小车落地，防止刚出生就被判坠崖
echo "⏳ 等待小车底盘落地 (3秒)..."
sleep 3

# -------------------------------------------------------------------------
# Tab 3: 唤醒动态行人
# -------------------------------------------------------------------------
gnome-terminal --tab --title="🚶 Pedestrian" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
echo '【阶段 3】正在注入动态障碍物逻辑...';
python3 scripts/move_pedestrian.py;
exec bash"

# -------------------------------------------------------------------------
# Tab 4: 启动 AI 大脑与裁判系统
# -------------------------------------------------------------------------
gnome-terminal --tab --title="🧠 RL Train" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
echo '【阶段 4】正在连接强化学习神经网络...';
python3 vision_track/train_vision.py;
exec bash"

echo "✅ 【系统】所有核心节点已成功发射！请在弹出的终端标签页中监控运行状态。"

# -------------------------------------------------------------------------
# Tab 5: open the monitor dashboard
# -------------------------------------------------------------------------
gnome-terminal --tab --title="Monitor dashboard" -- bash -c "
cd ${PROJECT_ROOT};
tensorboard --logdir=./logs/tensorboard/;
echo '【阶段 5】Open the monitor dashboard..., please vist http://localhost:6006';
exec bash"