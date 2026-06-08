from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory("mlp_astar_planner")

    map_yaml = os.path.join(
        pkg_share,
        "maps",
        "map_large.yaml",
    )

    rviz_config = os.path.join(
        pkg_share,
        "rviz",
        "mlp_astar.rviz",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "yaml_filename": map_yaml,
            }
        ],
    )

    lifecycle_bringup = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "nav2_util",
                    "lifecycle_bringup",
                    "map_server",
                ],
                output="screen",
            )
        ],
    )

    points = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="mlp_astar_planner",
                executable="points",
                name="points_publisher",
                output="screen",
            )
        ],
    )

    astar_mlp = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="mlp_astar_planner",
                executable="astar_mlp",
                name="mlp_astar_full_node",
                output="screen",
            )
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            rviz_config,
        ],
        output="screen",
    )

    print(f"Map yaml: {map_yaml}")
    print(f"Map exists: {os.path.exists(map_yaml)}")
    print(f"RViz config: {rviz_config}")
    print(f"RViz exists: {os.path.exists(rviz_config)}")

    return LaunchDescription(
        [
            map_server,
            lifecycle_bringup,
            points,
            astar_mlp,
            rviz,
        ]
    )