# MLP A* Planner — Launch Instructions

## 1. Go to your ROS 2 workspace

For example:

```bash
cd ~/ros2_ws
```

## 2. Clone the repository

Go to the `src` directory and clone the project:

```bash
cd src
git clone https://github.com/krukich/MiAPR-project.git mlp_astar_planner
cd ..
```

## 3. Install requirements

Make sure that PyTorch is installed:

```bash
pip install torch
```

## 4. Build the workspace

```bash
colcon build --symlink-install
```

## 5. Source the workspace

```bash
source install/setup.bash
```

## 6. Launch the project

```bash
ros2 launch mlp_astar_planner astar_comparision.launch.py
```
