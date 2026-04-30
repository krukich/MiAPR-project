Go to your ROS2 workspase directory, f.e. ros2_ws.

Commands to launch project:
cd src
git clone https://github.com/krukich/MiAPR-project.git mlp_astar_planner
cd ..
colcon build --symlink-install
source install/setup.bash
ros2 launch mlp_astar_planner astar_comparision.launch.py
