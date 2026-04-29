from copy import deepcopy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker
import time
from rclpy import qos


class GridMap(Node):
    def __init__(self, node_name='graph_search'):
        super().__init__(node_name)
        self.map = None
        self.start = None
        self.end = None

        qos_profile = qos.QoSProfile(depth=10)
        qos_profile.durability = qos.DurabilityPolicy.TRANSIENT_LOCAL

        self.sub_map = self.create_subscription(OccupancyGrid, 'map', self.map_callback, qos_profile)
        self.sub_start_pt = self.create_subscription(Marker, 'point_start', self.set_start, 10)
        self.sub_end_pt = self.create_subscription(Marker, 'point_end', self.set_end, 10)

        self.pub_map = self.create_publisher(OccupancyGrid, 'map_visited', 10)
        self.pub_path = self.create_publisher(Path, 'path', 10)

        self.get_logger().info("Object initialized!")

    def data_received(self):
        return self.map is not None and self.start is not None and self.end is not None

    def map_callback(self, data: OccupancyGrid):
        self.map = deepcopy(data)
        self.map.data = list(self.map.data)

    def get_marker_xy(self, marker: Marker):
        while self.map is None:
            time.sleep(0.1)

        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y

        x = int((marker.pose.position.x - ox) / res)
        y = int((marker.pose.position.y - oy) / res)
        return x, y

    def set_start(self, data: Marker):
        self.start = self.get_marker_xy(data)

    def set_end(self, data: Marker):
        self.end = self.get_marker_xy(data)

    def publish_visited(self, delay=0.0):
        self.pub_map.publish(self.map)

    def publish_path(self, path: list):
        path_msg = Path()
        path_msg.header.frame_id = 'map'

        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y

        for p in path:
            pose = PoseStamped()
            pose.pose.position.x = ox + res * (p[0] + 0.5)
            pose.pose.position.y = oy + res * (p[1] + 0.5)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses.append(pose)

        self.pub_path.publish(path_msg)

    def search(self):
        return NotImplementedError()