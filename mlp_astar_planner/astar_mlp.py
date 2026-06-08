#!/usr/bin/env python3

import os
import time
import heapq as pq

import rclpy
import torch
import numpy as np

from rclpy import qos
from ament_index_python.packages import get_package_share_directory

from mlp_astar_planner.grid_map import GridMap
from mlp_astar_planner.mlp_model import OccupancyMLP
from mlp_astar_planner.mlp_gradient_viz import MLPGradientVisualizer


class MLPAstar(GridMap):
    def __init__(self):
        super().__init__("mlp_astar_full_node")

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_path = self.create_publisher(
            self.pub_path.msg_type,
            "path_mlp_full",
            qos_profile,
        )

        self.pub_map = self.create_publisher(
            self.pub_map.msg_type,
            "map_visited_mlp_full",
            qos_profile,
        )

        self.lambda_grad = 4.0
        self.occupancy_threshold = 0.5

        self.model = OccupancyMLP()
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location="cpu"))
        self.model.eval()

        self.gradient_viz = MLPGradientVisualizer(
            self,
            self.model,
            "/mlp_full",
            stride=6,
            show_away_direction=False,
        )

        self.get_logger().info("MLP model loaded")
        self.get_logger().info("Experiment 2: full MLP A*")

    def get_model_path(self):
        pkg_share = get_package_share_directory("mlp_astar_planner")

        installed_model = os.path.join(
            pkg_share,
            "data",
            "mlp_model.pth",
        )

        if os.path.exists(installed_model):
            return installed_model

        workspace_dir = os.path.abspath(
            os.path.join(pkg_share, "..", "..", "..", "..")
        )

        source_model = os.path.join(
            workspace_dir,
            "src",
            "mlp_astar_planner",
            "data",
            "mlp_model.pth",
        )

        return source_model

    def normalize_cell(self, x, y):
        width = self.map.info.width
        height = self.map.info.height

        x_norm = x / max(width - 1, 1)
        y_norm = (height - 1 - y) / max(height - 1, 1)

        return x_norm, y_norm

    def query_network(self, x, y):
        x_norm, y_norm = self.normalize_cell(x, y)

        inp = torch.tensor(
            [[x_norm, y_norm]],
            dtype=torch.float32,
            requires_grad=True,
        )

        logit = self.model(inp)
        prob = torch.sigmoid(logit)

        grad = torch.autograd.grad(
            prob,
            inp,
            retain_graph=False,
            create_graph=False,
        )[0]

        grad_np = grad.detach().cpu().numpy()[0]
        grad_norm = float(np.linalg.norm(grad_np))

        return float(prob.item()), grad_norm

    def heuristics(self, pos):
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])

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

    def publish_visited_layer(self, visited_map_data):
        old_map_data = self.map.data

        self.map.data = visited_map_data
        self.publish_visited()

        self.map.data = old_map_data

    def reconstruct_path(self, came_from, start, goal):
        path = []
        node = goal

        while node != start:
            path.append(node)
            node = came_from[node]

        path.append(start)
        path.reverse()

        return path

    def search(self):
        self.gradient_viz.publish(self.map)

        width = self.map.info.width
        height = self.map.info.height

        map_cell_count = width * height
        visited_map_data = [0] * map_cell_count

        def to_index(x, y):
            return y * width + x

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

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
            _, current_g, current = pq.heappop(open_list)

            if current in visited:
                continue

            visited.add(current)

            if current != start and current != goal:
                idx = to_index(current[0], current[1])
                visited_map_data[idx] = 50

            if current == goal:
                path = self.reconstruct_path(came_from, start, goal)

                planning_time = time.perf_counter() - t0
                path_length_cells = len(path) - 1
                path_length_m = self.compute_path_length_m(path)

                self.publish_visited_layer(visited_map_data)
                self.publish_path(path)

                self.get_logger().info(
                    f"Full MLP A* path found | "
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

                if neighbor in visited:
                    continue

                prob, grad_norm = self.query_network(nx, ny)

                if prob > self.occupancy_threshold:
                    continue

                gradient_penalty = self.lambda_grad * grad_norm
                tentative_g = current_g + move_cost + gradient_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    f_score = tentative_g + self.heuristics(neighbor)
                    pq.heappush(open_list, (f_score, tentative_g, neighbor))

        planning_time = time.perf_counter() - t0

        self.publish_visited_layer(visited_map_data)

        self.get_logger().warn(
            f"Full MLP A* path not found | "
            f"time={planning_time:.4f}s | "
            f"visited={len(visited)}"
        )


def main(args=None):
    rclpy.init(args=args)

    planner = MLPAstar()

    while not planner.data_received():
        planner.get_logger().info("Waiting for map, start and goal...")
        rclpy.spin_once(planner)
        time.sleep(0.5)

    planner.get_logger().info("Start full MLP A* planning")
    planner.search()

    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()