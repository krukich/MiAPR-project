import math

import numpy as np
import torch

from rclpy import qos
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class MLPGradientVisualizer:
    def __init__(
        self,
        node,
        model,
        topic_prefix,
        stride=10,
        batch_size=4096,
        arrow_len_cells=2.0,
        show_away_direction=False,
    ):
        self.node = node
        self.model = model
        self.topic_prefix = topic_prefix
        self.stride = stride
        self.batch_size = batch_size
        self.arrow_len_cells = arrow_len_cells
        self.show_away_direction = show_away_direction

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL
        qos_profile.reliability = qos.ReliabilityPolicy.RELIABLE

        self.pub_occupancy = node.create_publisher(
            OccupancyGrid,
            f"{topic_prefix}/occupancy_heatmap",
            qos_profile,
        )

        self.pub_gradient = node.create_publisher(
            OccupancyGrid,
            f"{topic_prefix}/gradient_heatmap",
            qos_profile,
        )

        self.pub_vectors = node.create_publisher(
            MarkerArray,
            f"{topic_prefix}/gradient_vectors",
            qos_profile,
        )

    def publish(self, map_msg):
        width = map_msg.info.width
        height = map_msg.info.height

        if width == 0 or height == 0:
            self.node.get_logger().warn(
                f"Cannot publish MLP heatmaps for {self.topic_prefix}: empty map"
            )
            return

        cells = self.make_cells(width, height)
        probs, gx, gy, grad_mag = self.compute_mlp_fields(cells, width, height)

        self.publish_occupancy_heatmap(map_msg, probs)
        self.publish_gradient_heatmap(map_msg, grad_mag)
        self.publish_vectors(map_msg, cells, gx, gy, grad_mag)

        self.node.get_logger().info(
            f"Published full-map MLP visualization: "
            f"{self.topic_prefix}/occupancy_heatmap, "
            f"{self.topic_prefix}/gradient_heatmap, "
            f"{self.topic_prefix}/gradient_vectors"
        )

    def make_cells(self, width, height):
        yy, xx = np.mgrid[0:height, 0:width]
        cells = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return cells.astype(np.float32)

    def compute_mlp_fields(self, cells, width, height):
        all_probs = []
        all_grads = []

        for start in range(0, len(cells), self.batch_size):
            batch = cells[start:start + self.batch_size]

            normalized = np.zeros((len(batch), 2), dtype=np.float32)
            normalized[:, 0] = batch[:, 0] / max(width - 1, 1)
            normalized[:, 1] = (height - 1 - batch[:, 1]) / max(height - 1, 1)

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

            all_probs.append(probs.detach().cpu().numpy())
            all_grads.append(grads.detach().cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        grads = np.concatenate(all_grads, axis=0)

        gx = grads[:, 0] / max(width - 1, 1)
        gy = -grads[:, 1] / max(height - 1, 1)

        grad_mag = np.sqrt(gx * gx + gy * gy)

        return probs, gx, gy, grad_mag

    def publish_occupancy_heatmap(self, map_msg, probs):
        occ = np.clip(probs * 100.0, 0.0, 100.0).astype(np.int8)

        grid = OccupancyGrid()
        grid.header.stamp = self.node.get_clock().now().to_msg()
        grid.header.frame_id = map_msg.header.frame_id if map_msg.header.frame_id else "map"
        grid.info = map_msg.info
        grid.data = occ.tolist()

        self.pub_occupancy.publish(grid)

    def normalize_gradient_heatmap(self, grad_mag):
        nonzero = grad_mag[grad_mag > 1e-12]

        if len(nonzero) == 0:
            return np.zeros_like(grad_mag, dtype=np.int8)

        scale = np.percentile(nonzero, 98)

        if scale <= 1e-12:
            scale = float(np.max(nonzero))

        if scale <= 1e-12:
            return np.zeros_like(grad_mag, dtype=np.int8)

        heat = np.clip((grad_mag / scale) * 100.0, 0.0, 100.0)

        return heat.astype(np.int8)

    def publish_gradient_heatmap(self, map_msg, grad_mag):
        heat = self.normalize_gradient_heatmap(grad_mag)

        grid = OccupancyGrid()
        grid.header.stamp = self.node.get_clock().now().to_msg()
        grid.header.frame_id = map_msg.header.frame_id if map_msg.header.frame_id else "map"
        grid.info = map_msg.info
        grid.data = heat.tolist()

        self.pub_gradient.publish(grid)

    def publish_vectors(self, map_msg, cells, gx, gy, grad_mag):
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin

        heat = self.normalize_gradient_heatmap(grad_mag)

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0

        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                idx = y * width + x

                if heat[idx] < 12:
                    continue

                vx = float(gx[idx])
                vy = float(gy[idx])

                if self.show_away_direction:
                    vx = -vx
                    vy = -vy

                norm = math.sqrt(vx * vx + vy * vy)

                if norm < 1e-12:
                    continue

                vx /= norm
                vy /= norm

                strength = float(heat[idx]) / 100.0
                arrow_len = self.arrow_len_cells * resolution * (0.4 + 0.6 * strength)

                p1 = Point()
                p1.x = origin.position.x + (x + 0.5) * resolution
                p1.y = origin.position.y + (y + 0.5) * resolution
                p1.z = 0.04

                p2 = Point()
                p2.x = p1.x + vx * arrow_len
                p2.y = p1.y + vy * arrow_len
                p2.z = 0.04

                marker = Marker()
                marker.header.stamp = self.node.get_clock().now().to_msg()
                marker.header.frame_id = map_msg.header.frame_id if map_msg.header.frame_id else "map"
                marker.ns = f"{self.topic_prefix.strip('/')}_gradient_vectors"
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.points = [p1, p2]

                marker.scale.x = resolution * 0.18
                marker.scale.y = resolution * 0.40
                marker.scale.z = resolution * 0.40

                marker.color.r = 1.0
                marker.color.g = 0.15
                marker.color.b = 0.0
                marker.color.a = 0.85

                marker_array.markers.append(marker)
                marker_id += 1

        self.pub_vectors.publish(marker_array)