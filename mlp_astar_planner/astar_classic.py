#!/usr/bin/env python3

import time
import heapq as pq
from rclpy import qos
import rclpy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from mlp_astar_planner.grid_map import GridMap


class ClassicAstar(GridMap):
    def __init__(self):
        super().__init__("classic_astar_node")

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_path_classic = self.create_publisher(Path, "path_classic", qos_profile)

    def heuristics(self, pos):
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])

    def search(self):
        width = self.map.info.width
        height = self.map.info.height

        original_data = list(self.map.data)

        def to_index(x, y):
            return y * width + x

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def is_wall(x, y):
            return original_data[to_index(x, y)] == 100

        start = tuple(self.start)
        goal = tuple(self.end)

        open_list = []
        pq.heappush(open_list, (self.heuristics(start), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]

        t0 = time.perf_counter()

        while open_list:
            _, _, current = pq.heappop(open_list)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                path = []
                node = current

                while node != start:
                    path.append(node)
                    node = came_from[node]

                path.append(start)
                path.reverse()

                planning_time = time.perf_counter() - t0
                path_length_cells = len(path) - 1
                path_length_m = self.compute_path_length_m(path)

                self.publish_classic_path(path)

                self.get_logger().info(
                    f"Classic A* path found | "
                    f"time={planning_time:.4f}s | "
                    f"visited={len(visited)} | "
                    f"path_cells={path_length_cells} | "
                    f"path_m={path_length_m:.3f}"
                )
                return

            for dx, dy, move_cost in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                neighbor = (nx, ny)

                if not in_bounds(nx, ny):
                    continue

                if is_wall(nx, ny):
                    continue

                if neighbor in visited:
                    continue

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self.heuristics(neighbor)
                    pq.heappush(open_list, (f_score, tentative_g, neighbor))

        planning_time = time.perf_counter() - t0

        self.get_logger().warn(
            f"Classic A* path not found | "
            f"time={planning_time:.4f}s | "
            f"visited={len(visited)}"
        )

    def publish_classic_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y

        for p in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = ox + res * (p[0] + 0.5)
            pose.pose.position.y = oy + res * (p[1] + 0.5)
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.pub_path_classic.publish(path_msg)

    def compute_path_length_m(self, path):
        if len(path) < 2:
            return 0.0

        res = self.map.info.resolution
        length_cells = 0.0

        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            length_cells += (dx * dx + dy * dy) ** 0.5

        return length_cells * res


def main(args=None):
    rclpy.init(args=args)

    planner = ClassicAstar()

    while not planner.data_received():
        planner.get_logger().info("Waiting for map, start and goal...")
        rclpy.spin_once(planner)
        time.sleep(0.5)

    planner.get_logger().info("Start classic A* planning")
    planner.search()

    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()