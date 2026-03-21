#!/bin/bash
PROJECT_ROOT=$(pwd)
echo "🎬 【系统】正在启动 BEV 降维终极架构 - 阅兵演示模式..."

# 🛡️ 极其关键的第一道防线：阅兵前绝对清理战场，防止车子因为旧进程而乱飞
echo "🧹 正在清理后台残留的 Gazebo 僵尸进程..."
killall -9 gzserver gzclient > /dev/null 2>&1
sleep 2

# Tab 1: Gazebo (恢复 GUI 界面，专为录像准备)
gnome-terminal --tab --title="🌍 Gazebo GUI" -- bash -c "
source /opt/ros/humble/setup.bash;
echo '【系统】正在启动带界面的 Gazebo 物理引擎...';
ros2 launch gazebo_ros gazebo.launch.py world:=${PROJECT_ROOT}/worlds/custom_highway.world;
exec bash"

# 给 GUI 预留充足的启动时间
echo "⏳ 等待物理世界加载 (8秒)..."
sleep 8

# Tab 2: 生成小车
gnome-terminal --tab --title="🚗 Spawn Car" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle_pi;
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py x_pose:=0.2 y_pose:=0.0 z_pose:=0.05;
exec bash"
sleep 3

# Tab 3: 生成行人干扰
gnome-terminal --tab --title="🚶 Pedestrian" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
python3 scripts/move_pedestrian.py;
exec bash"

# Tab 4: 挂载 RL 极品大脑 (纯推理模式)
gnome-terminal --tab --title="🧠 BEV Eval" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
echo '🔥 【系统】正在挂载巅峰脑电波，开始自动驾驶阅兵！';
python3 bev_track/eval.py;
exec bash"