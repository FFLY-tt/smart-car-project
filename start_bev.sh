#!/bin/bash
PROJECT_ROOT=$(pwd)
echo "🚀 【系统】正在启动 BEV 降维终极架构仿真环境..."

# Tab 1: Gazebo
gnome-terminal --tab --title="🌍 Gazebo World" -- bash -c "
source /opt/ros/humble/setup.bash;
ros2 launch gazebo_ros gazebo.launch.py world:=${PROJECT_ROOT}/worlds/custom_highway.world;
exec bash"
sleep 5

# Tab 2: 小车
gnome-terminal --tab --title="🚗 Spawn Car" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle_pi;
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py x_pose:=0.2 y_pose:=0.0 z_pose:=0.05;
exec bash"
sleep 3

# Tab 3: 行人
gnome-terminal --tab --title="🚶 Pedestrian" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
python3 scripts/move_pedestrian.py;
exec bash"

# Tab 4: 启动新的 BEV 训练大脑
gnome-terminal --tab --title="🧠 BEV RL Train" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
python3 bev_track/train_bev.py;
exec bash"

# Tab 5: 公共监控大屏
gnome-terminal --tab --title="Monitor dashboard" -- bash -c "
cd ${PROJECT_ROOT};
tensorboard --logdir=./logs/tensorboard/ --port=6006;
exec bash"