import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import csv
import math
import os


class GlobalPathPublisher(Node):

    def __init__(self):
        super().__init__('global_path_publisher')

        self.publisher = self.create_publisher(
            Path,
            '/global_path',
            10
        )

        # [수정] 이 파일과 같은 폴더의 global_path.csv (어느 컴퓨터에 clone 해도 동작)
        self.csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_path.csv')

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.load_path()

        self.timer = self.create_timer(1.0, self.publish_path)

        self.get_logger().info(
            f'Global path loaded: {len(self.path_msg.poses)} points'
        )

    def load_path(self):

        points = []

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 2:
                    continue

                x = float(row[0])
                y = float(row[1])

                points.append((x, y))

        for i, (x, y) in enumerate(points):

            pose = PoseStamped()

            pose.header.frame_id = 'map'

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # 경로 진행 방향 계산
            if i < len(points) - 1:
                next_x, next_y = points[i + 1]
            else:
                next_x, next_y = points[0]

            yaw = math.atan2(
                next_y - y,
                next_x - x
            )

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            self.path_msg.poses.append(pose)

    def publish_path(self):

        now = self.get_clock().now().to_msg()

        self.path_msg.header.stamp = now

        for pose in self.path_msg.poses:
            pose.header.stamp = now

        self.publisher.publish(self.path_msg)


def main(args=None):

    rclpy.init(args=args)

    node = GlobalPathPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
