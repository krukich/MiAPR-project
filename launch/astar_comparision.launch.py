from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("mlp_astar_planner")

    workspace_dir = os.path.abspath(
        os.path.join(pkg_share, "..", "..", "..", "..")
    )

    package_src_dir = os.path.join(
        workspace_dir,
        "src",
        "mlp_astar_planner"
    )

    rviz_config = os.path.join(
        package_src_dir,
        "rviz",
        "astar_comparision.rviz"
    )

    map_yaml = os.path.join(
        package_src_dir,
        "maps",
        "map_large.yaml"
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"yaml_filename": map_yaml}]
    )

    lifecycle_bringup = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "run", "nav2_util", "lifecycle_bringup", "map_server"],
                output="screen"
            )
        ]
    )

    points = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="mlp_astar_planner",
                executable="points",
                name="points_publisher",
                output="screen"
            )
        ]
    )

    astar_classic = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="mlp_astar_planner",
                executable="astar_classic",
                name="classic_astar_node",
                output="screen"
            )
        ]
    )

    astar_mlp = TimerAction(
        period=5.5,
        actions=[
            Node(
                package="mlp_astar_planner",
                executable="astar_mlp",
                name="mlp_astar_node",
                output="screen"
            )
        ]
    )

    print(f"RViz config: {rviz_config}")
    print(f"Exists: {os.path.exists(rviz_config)}")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen"
    )

    return LaunchDescription([
        map_server,
        lifecycle_bringup,
        points,
        astar_classic,
        astar_mlp,
        rviz,
    ])