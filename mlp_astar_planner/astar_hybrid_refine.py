#!/usr/bin/env python3

import os
import time
import heapq as pq

import rclpy
import torch
import numpy as np

from rclpy import qos
from nav_msgs.msg import OccupancyGrid, Path
from ament_index_python.packages import get_package_share_directory

from mlp_astar_planner.grid_map import GridMap
from mlp_astar_planner.mlp_model import OccupancyMLP
from mlp_astar_planner.mlp_gradient_viz import MLPGradientVisualizer


class HybridRefineAstar(GridMap):
    def __init__(self):
        super().__init__("hybrid_refine_astar_node")

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_path = self.create_publisher(
            Path,
            "path_mlp_refined",
            qos_profile,
        )

        self.pub_map = self.create_publisher(
            OccupancyGrid,
            "map_visited_mlp_refined",
            qos_profile,
        )

        self.occupancy_threshold = 0.5

        self.wall_radius_cells = 3
        self.refine_iterations = 8
        self.refine_step_cells = 0.9
        self.smooth_weight = 0.15

        self.model = OccupancyMLP()
        self.model.load_state_dict(torch.load(self.get_model_path(), map_location="cpu"))
        self.model.eval()

        self.gradient_viz = MLPGradientVisualizer(
            self,
            self.model,
            "/mlp_refined",
            stride=6,
            show_away_direction=False,
        )

        self.get_logger().info("MLP model loaded")
        self.get_logger().info("Experiment 3: classic A* + MLP point refinement, 4-connected final path")

    def get_model_path(self):
        pkg_share = get_package_share_directory("mlp_astar_planner")
        workspace_dir = os.path.abspath(
            os.path.join(pkg_share, "..", "..", "..", "..")
        )

        return os.path.join(
            workspace_dir,
            "src",
            "mlp_astar_planner",
            "data",
            "mlp_model.pth",
        )

    def heuristics(self, pos):
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])

    def to_index(self, x, y, width):
        return y * width + x

    def in_bounds(self, x, y, width, height):
        return 0 <= x < width and 0 <= y < height

    def is_wall(self, data, x, y, width, height):
        x = int(round(x))
        y = int(round(y))

        if not self.in_bounds(x, y, width, height):
            return True

        value = data[self.to_index(x, y, width)]
        return value >= 50

    def is_free(self, data, x, y, width, height):
        x = int(round(x))
        y = int(round(y))

        if not self.in_bounds(x, y, width, height):
            return False

        value = data[self.to_index(x, y, width)]
        return value == 0

    def reconstruct_path(self, came_from, start, goal):
        path = []
        node = goal

        while node != start:
            path.append(node)
            node = came_from[node]

        path.append(start)
        path.reverse()

        return path

    def classic_astar_path(self, original_data):
        width = self.map.info.width
        height = self.map.info.height

        start = tuple(self.start)
        goal = tuple(self.end)

        open_list = []
        pq.heappush(open_list, (self.heuristics(start), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        visited_data = list(original_data)

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
                idx = self.to_index(current[0], current[1], width)
                if visited_data[idx] == 0:
                    visited_data[idx] = 50

            if current == goal:
                planning_time = time.perf_counter() - t0
                path = self.reconstruct_path(came_from, start, goal)
                return path, visited, visited_data, planning_time

            for dx, dy, move_cost in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                neighbor = (nx, ny)

                if not self.in_bounds(nx, ny, width, height):
                    continue

                if self.is_wall(original_data, nx, ny, width, height):
                    continue

                if neighbor in visited:
                    continue

                tentative_g = current_g + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    f_score = tentative_g + self.heuristics(neighbor)
                    pq.heappush(open_list, (f_score, tentative_g, neighbor))

        planning_time = time.perf_counter() - t0
        return None, visited, visited_data, planning_time

    def query_network_batch(self, points):
        width = self.map.info.width
        height = self.map.info.height

        if not points:
            return np.array([]), np.empty((0, 2), dtype=np.float32)

        normalized = []

        for x, y in points:
            x_norm = x / max(width - 1, 1)
            y_norm = (height - 1 - y) / max(height - 1, 1)
            normalized.append([x_norm, y_norm])

        inp = torch.tensor(
            normalized,
            dtype=torch.float32,
            requires_grad=True,
        )

        logits = self.model(inp)
        probs = torch.sigmoid(logits).squeeze(-1)

        grads = torch.autograd.grad(
            probs.sum(),
            inp,
            retain_graph=False,
            create_graph=False,
        )[0]

        probs_np = probs.detach().cpu().numpy()
        grads_np = grads.detach().cpu().numpy()

        return probs_np, grads_np

    def network_away_direction(self, grad):
        width = self.map.info.width
        height = self.map.info.height

        cell_grad = np.array(
            [
                grad[0] / max(width - 1, 1),
                -grad[1] / max(height - 1, 1),
            ],
            dtype=np.float32,
        )

        norm = np.linalg.norm(cell_grad)

        if norm < 1e-12:
            return np.zeros(2, dtype=np.float32)

        return -cell_grad / norm

    def map_away_direction(self, data, x, y):
        width = self.map.info.width
        height = self.map.info.height

        cx = int(round(x))
        cy = int(round(y))

        force = np.zeros(2, dtype=np.float32)
        radius = self.wall_radius_cells

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                wx = cx + dx
                wy = cy + dy

                if dx * dx + dy * dy > radius * radius:
                    continue

                if not self.is_wall(data, wx, wy, width, height):
                    continue

                vx = x - wx
                vy = y - wy
                dist2 = vx * vx + vy * vy

                if dist2 < 1e-6:
                    continue

                force += np.array([vx, vy], dtype=np.float32) / dist2

        norm = np.linalg.norm(force)

        if norm < 1e-12:
            return np.zeros(2, dtype=np.float32)

        return force / norm

    def is_near_wall(self, data, x, y):
        width = self.map.info.width
        height = self.map.info.height

        cx = int(round(x))
        cy = int(round(y))

        radius = self.wall_radius_cells

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue

                wx = cx + dx
                wy = cy + dy

                if self.is_wall(data, wx, wy, width, height):
                    return True

        return False

    def is_segment_free(self, data, a, b):
        width = self.map.info.width
        height = self.map.info.height

        ax, ay = a
        bx, by = b

        steps = int(max(abs(bx - ax), abs(by - ay)) * 2) + 1
        steps = max(steps, 1)

        for i in range(steps + 1):
            t = i / steps
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)

            if not self.is_free(data, x, y, width, height):
                return False

        return True

    def clamp_point(self, point):
        width = self.map.info.width
        height = self.map.info.height

        point[0] = float(np.clip(point[0], 0.0, width - 1.0))
        point[1] = float(np.clip(point[1], 0.0, height - 1.0))

        return point

    def refine_path_with_mlp(self, path, original_data):
        if path is None or len(path) <= 2:
            return path, 0

        width = self.map.info.width
        height = self.map.info.height

        points = [np.array([float(x), float(y)], dtype=np.float32) for x, y in path]
        total_moved = 0

        for _ in range(self.refine_iterations):
            inner_points = [tuple(p) for p in points[1:-1]]
            _, grads = self.query_network_batch(inner_points)

            new_points = [points[0].copy()]
            moved_this_iter = 0

            for local_idx, i in enumerate(range(1, len(points) - 1)):
                current = points[i]
                previous = points[i - 1]
                next_point = points[i + 1]

                near_wall = self.is_near_wall(
                    original_data,
                    current[0],
                    current[1],
                )

                smooth_target = 0.5 * (previous + next_point)
                smooth_delta = self.smooth_weight * (smooth_target - current)

                if near_wall:
                    away = self.network_away_direction(grads[local_idx])

                    if np.linalg.norm(away) < 1e-8:
                        away = self.map_away_direction(
                            original_data,
                            current[0],
                            current[1],
                        )

                    candidate = current + self.refine_step_cells * away + smooth_delta
                else:
                    candidate = current + smooth_delta

                candidate = self.clamp_point(candidate)

                valid = (
                    self.is_free(original_data, candidate[0], candidate[1], width, height)
                    and self.is_segment_free(original_data, previous, candidate)
                    and self.is_segment_free(original_data, candidate, next_point)
                )

                if valid:
                    if np.linalg.norm(candidate - current) > 0.05:
                        moved_this_iter += 1

                    new_points.append(candidate)
                else:
                    new_points.append(current.copy())

            new_points.append(points[-1].copy())
            points = new_points
            total_moved += moved_this_iter

            if moved_this_iter == 0:
                break

        refined_path = [(float(p[0]), float(p[1])) for p in points]
        return refined_path, total_moved

    def snap_path_to_grid(self, path):
        snapped = []

        for x, y in path:
            cell = (int(round(x)), int(round(y)))

            if not snapped or snapped[-1] != cell:
                snapped.append(cell)

        return snapped

    def astar_between_cells(self, original_data, start, goal):
        width = self.map.info.width
        height = self.map.info.height

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        if start == goal:
            return [start]

        open_list = []
        pq.heappush(open_list, (abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]

        def h(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        while open_list:
            _, current_g, current = pq.heappop(open_list)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, start, goal)

            for dx, dy, move_cost in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                neighbor = (nx, ny)

                if not self.in_bounds(nx, ny, width, height):
                    continue

                if neighbor in visited:
                    continue

                if not self.is_free(original_data, nx, ny, width, height):
                    continue

                tentative_g = current_g + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    f_score = tentative_g + h(neighbor)
                    pq.heappush(open_list, (f_score, tentative_g, neighbor))

        return None

    def make_path_4_connected(self, path, original_data):
        if path is None or len(path) <= 1:
            return path

        fixed_path = [path[0]]

        for i in range(1, len(path)):
            previous = fixed_path[-1]
            current = path[i]

            dx = abs(current[0] - previous[0])
            dy = abs(current[1] - previous[1])

            if dx == 0 and dy == 0:
                continue

            if dx + dy == 1:
                fixed_path.append(current)
                continue

            connector = self.astar_between_cells(
                original_data,
                previous,
                current,
            )

            if connector is None:
                continue

            for cell in connector[1:]:
                if fixed_path[-1] != cell:
                    fixed_path.append(cell)

        return fixed_path

    def validate_4_connected_path(self, path):
        if path is None or len(path) <= 1:
            return True

        for i in range(1, len(path)):
            dx = abs(path[i][0] - path[i - 1][0])
            dy = abs(path[i][1] - path[i - 1][1])

            if dx + dy != 1:
                return False

        return True

    def compute_path_length_m(self, path):
        if path is None or len(path) < 2:
            return 0.0

        res = self.map.info.resolution
        length_cells = 0.0

        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            length_cells += (dx * dx + dy * dy) ** 0.5

        return length_cells * res

    def search(self):
        self.gradient_viz.publish(self.map)
        original_data = list(self.map.data)

        raw_path, visited, visited_data, astar_time = self.classic_astar_path(original_data)

        self.map.data = visited_data
        self.publish_visited()

        if raw_path is None:
            self.get_logger().warn(
                f"Hybrid refine path not found | "
                f"astar_time={astar_time:.4f}s | "
                f"visited={len(visited)}"
            )
            return

        t0 = time.perf_counter()

        refined_path, moved_points = self.refine_path_with_mlp(raw_path, original_data)
        refined_path = self.snap_path_to_grid(refined_path)
        refined_path = self.make_path_4_connected(refined_path, original_data)

        refine_time = time.perf_counter() - t0

        raw_length_m = self.compute_path_length_m(raw_path)
        refined_length_m = self.compute_path_length_m(refined_path)

        is_4_connected = self.validate_4_connected_path(refined_path)

        self.publish_path(refined_path)

        self.get_logger().info(
            f"Hybrid refine path found | "
            f"astar_time={astar_time:.4f}s | "
            f"refine_time={refine_time:.4f}s | "
            f"total_time={astar_time + refine_time:.4f}s | "
            f"visited={len(visited)} | "
            f"raw_cells={len(raw_path) - 1} | "
            f"refined_cells={len(refined_path) - 1} | "
            f"moved_points={moved_points} | "
            f"raw_m={raw_length_m:.3f} | "
            f"refined_m={refined_length_m:.3f} | "
            f"4_connected={is_4_connected}"
        )


def main(args=None):
    rclpy.init(args=args)

    planner = HybridRefineAstar()

    while not planner.data_received():
        planner.get_logger().info("Waiting for map, start and goal...")
        rclpy.spin_once(planner)
        time.sleep(0.5)

    planner.get_logger().info("Start hybrid refine A* + MLP planning")
    planner.search()

    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()