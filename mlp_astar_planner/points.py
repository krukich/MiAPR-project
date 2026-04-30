import random
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from rclpy import qos


class PointsPublisher(Node):
    def __init__(self):
        super().__init__('points_publisher')

        self.map = None
        self.start_marker = None
        self.end_marker = None
        self.points_generated = False

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.pub_start = self.create_publisher(Marker, 'point_start', 10)
        self.pub_end = self.create_publisher(Marker, 'point_end', 10)

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL
        self.sub_map = self.create_subscription(OccupancyGrid, 'map', self.map_callback, qos_profile)

        random.seed()

    def map_callback(self, msg: OccupancyGrid):
        if self.points_generated:
            return

        self.map = msg

        width = self.map.info.width
        height = self.map.info.height
        data = list(self.map.data)

        free_cells = []
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if data[idx] == 0:
                    free_cells.append((x, y))

        if len(free_cells) < 2:
            self.get_logger().error("Not enough free cells on the map!")
            return

        # Хотим, чтобы start и goal были не слишком близко
        min_manhattan = max(10, min(width, height) // 4)

        start_cell = None
        end_cell = None

        for _ in range(500):
            s = random.choice(free_cells)
            g = random.choice(free_cells)

            if s == g:
                continue

            dist = abs(s[0] - g[0]) + abs(s[1] - g[1])
            if dist >= min_manhattan:
                start_cell = s
                end_cell = g
                break

        if start_cell is None or end_cell is None:
            start_cell, end_cell = random.sample(free_cells, 2)

        self.start_marker = self.build_marker_from_cell(start_cell[0], start_cell[1], "start", (0.0, 1.0, 0.0))
        self.end_marker = self.build_marker_from_cell(end_cell[0], end_cell[1], "end", (1.0, 0.0, 0.0))

        self.points_generated = True

        self.get_logger().info(f"Random start: {start_cell}")
        self.get_logger().info(f"Random goal:  {end_cell}")

    def build_marker_from_cell(self, cell_x: int, cell_y: int, name: str, color: tuple):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = name
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y

        marker.pose.position.x = ox + (cell_x + 0.5) * res
        marker.pose.position.y = oy + (cell_y + 0.5) * res
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.06
        marker.scale.y = 0.06
        marker.scale.z = 0.02

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8

        return marker

    def timer_callback(self):
        if not self.points_generated:
            return

        self.start_marker.header.stamp = self.get_clock().now().to_msg()
        self.end_marker.header.stamp = self.get_clock().now().to_msg()

        self.pub_start.publish(self.start_marker)
        self.pub_end.publish(self.end_marker)


def main(args=None):
    rclpy.init(args=args)
    points_publisher = PointsPublisher()
    rclpy.spin(points_publisher)
    points_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()