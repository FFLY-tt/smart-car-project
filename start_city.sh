#!/bin/bash
PROJECT_ROOT=$(pwd)
echo "🚀 【系统】正在启动 city仿真环境..."

# 🧹 启动前清理历史进程，防止 Gazebo 端口冲突导致黑屏
killall -9 gzserver gzclient > /dev/null 2>&1
sleep 1

# Tab 1: Gazebo
gnome-terminal --tab --title="🌍 Gazebo World" -- bash -c "
source /opt/ros/humble/setup.bash;
echo '【阶段 1】正在加载物理引擎与城市街区地图...';
# 👑 修正点 1：加载你用 Python 生成的最新地图 custom_city_g.world
ros2 launch gazebo_ros gazebo.launch.py world:=${PROJECT_ROOT}/worlds/custom_city_g.world;
exec bash"
sleep 4 # 稍微多给 Gazebo 一秒钟加载地图的时间

# Tab 2: 小车
gnome-terminal --tab --title="🚗 Spawn Car" -- bash -c "
source /opt/ros/humble/setup.bash;
export TURTLEBOT3_MODEL=waffle_pi;
echo '【阶段 2】正在起点 (X=9.6, Y=6.6) 空降 Turtlebot3 并强制车头朝北...';
MODEL_PATH=\$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf;
ros2 run gazebo_ros spawn_entity.py -entity waffle_pi -file \$MODEL_PATH -x 9.6 -y 6.6 -z 0.05 -Y 1.5708;

exec bash"
sleep 3

# Tab 3: 行人
gnome-terminal --tab --title="🚶 Pedestrian" -- bash -c "
source /opt/ros/humble/setup.bash;
cd ${PROJECT_ROOT};
echo '【阶段 3】挂载动态行人控制系统...';
python3 city_track/pedestrian_controller.py;
exec bash"