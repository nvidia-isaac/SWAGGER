# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from swagger_interfaces.msg import RouteRequest, RouteResult
from swagger_interfaces.srv import GenerateRoute
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from swagger.models import Point as SWAGGERPoint
from swagger.utils import pixel_to_world, world_to_pixel
from swagger.waypoint_graph_generator import (
    WaypointGraphGenerator,
    WaypointGraphGeneratorConfig,
)


def waypoints_to_cells(x0, y0, x1, y1):
    """Bresenham's line algorithm to generate cells in the occupancy grid between two points"""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


class RoutePlanner(Node):
    """
    Route Planner Node

    Uses the waypoint graph generator to find a route between any two points in the map.
    It runs on demand from either:
    - A service call (blocking)
    - A message received on the goal_topic (non-blocking)

    It maintains the robot's current position using TF, which is only updated when a new goal
    message is received.
    """

    def __init__(self, node_name, *, namespace=None, enable_rosout=True):
        super().__init__(node_name, namespace=namespace, enable_rosout=enable_rosout)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_file", rclpy.Parameter.Type.STRING),
                ("service_name", "generate_route"),
                ("route_frame_id", "map"),
                ("robot_frame_id", "base_link"),
                ("goal_topic", "goal_pose"),
                ("route_topic", "route"),
                ("route_result_topic", "route_result"),
                ("route_request_topic", "route_request"),
                ("swagger_viz_topic", "swagger_viz"),
                ("resolution", 0.05),
                ("safety_distance", 0.3),
                ("occupancy_threshold", 127),
                ("x_offset", 0.0),
                ("y_offset", 0.0),
                ("rotation", 0.0),
            ],
        )

        self._logger = self.get_logger()
        self._logger.info(f"Initialized {[self.get_name()]} with parameters as:")
        for param in self.get_parameters_by_prefix(""):
            self._logger.info(f"\t{param}: {self.get_parameter(param).value}")

        self._occupancy_map_file = self.get_parameter("map_file").get_parameter_value().string_value
        self._route_frame_id = self.get_parameter("route_frame_id").get_parameter_value().string_value
        self._robot_frame_id = self.get_parameter("robot_frame_id").get_parameter_value().string_value

        self.create_service(GenerateRoute, self.get_parameter("service_name").value, self._generate_route_callback)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self._new_goal, 10)
        self.create_subscription(
            RouteRequest, self.get_parameter("route_request_topic").value, self._route_request_callback, 10
        )
        self._pub_route = self.create_publisher(Path, self.get_parameter("route_topic").value, 10)
        self._pub_swagger_viz = self.create_publisher(MarkerArray, self.get_parameter("swagger_viz_topic").value, 10)
        self._pub_route_result = self.create_publisher(RouteResult, self.get_parameter("route_result_topic").value, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._initialize_swagger()

    def _initialize_swagger(self):
        self._swagger_config = WaypointGraphGeneratorConfig()
        self._swagger = WaypointGraphGenerator(config=self._swagger_config)

        map_image = cv2.imread(self._occupancy_map_file, cv2.IMREAD_GRAYSCALE)
        if map_image is None:
            raise ValueError(f"Failed to load occupancy map from {self._occupancy_map_file}")

        self._swagger.build_graph_from_grid_map(
            image=map_image,
            resolution=self.get_parameter("resolution").value,
            safety_distance=self.get_parameter("safety_distance").value,
            occupancy_threshold=self.get_parameter("occupancy_threshold").value,
            x_offset=self.get_parameter("x_offset").value,
            y_offset=self.get_parameter("y_offset").value,
            rotation=self.get_parameter("rotation").value,
        )
        self._cos_rot = np.cos(self.get_parameter("rotation").value)
        self._sin_rot = np.sin(self.get_parameter("rotation").value)

        self._draw_and_visualize_swagger()

    def _generate_route_callback(self, request, response):
        self._logger.debug(f"Generating route from: {request.start} to: {request.goal}")

        start_point = (request.start.x, request.start.y, request.start.z)
        goal_point = (request.goal.x, request.goal.y, request.goal.z)

        maybe_path = self._get_route(start_point, goal_point)
        if maybe_path is None:
            response.success = False
            response.message = "Failed to generate path"
        else:
            response.success = True
            response.message = "Path generated successfully"
            response.path = self._route_to_path_msg(maybe_path)
        return response

    def _route_request_callback(self, msg: RouteRequest) -> None:
        self._logger.debug(f"Route request received: {msg.start} to: {msg.goal}")
        start_point = (msg.start.x, msg.start.y, msg.start.z)
        goal_point = (msg.goal.x, msg.goal.y, msg.goal.z)
        maybe_path = self._get_route(start_point, goal_point)
        if maybe_path is None:
            result_msg = RouteResult()
            result_msg.route_id = msg.route_id
            result_msg.success = False
        else:
            result_msg = RouteResult()
            result_msg.route_id = msg.route_id
            result_msg.success = True
            result_msg.path = self._route_to_path_msg(maybe_path)
        self._pub_route_result.publish(result_msg)

    def _get_route(
        self,
        start_point: tuple[float, float, float],
        goal_point: tuple[float, float, float],
    ) -> Path | None:
        start_p = SWAGGERPoint(x=start_point[0], y=start_point[1], z=start_point[2])
        goal_p = SWAGGERPoint(x=goal_point[0], y=goal_point[1], z=goal_point[2])
        route = self._swagger.find_route(start_p, goal_p)
        if not route:
            self._logger.error(f"Failed to get route for start: {start_point} and goal: {goal_point}")
            return None
        return route

    def _route_to_path_msg(self, route: list[SWAGGERPoint]) -> Path:
        route_cells = self._transform_route_wps_cells(route)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self._route_frame_id
        for cell in route_cells:
            point = pixel_to_world(
                cell[0],
                cell[1],
                self.get_parameter("resolution").value,
                self.get_parameter("x_offset").value,
                self.get_parameter("y_offset").value,
                self._cos_rot,
                self._sin_rot,
            )

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.w = 1.0  # Default orientation
            path_msg.poses.append(pose)
        return path_msg

    def _transform_route_wps_cells(self, route_wps: list[SWAGGERPoint]) -> list[tuple[int, int]]:
        # Use Bresenham's line algorithm to convert waypoints in world coordinates to cells in the occupancy grid
        route_cells = []
        for i in range(len(route_wps) - 1):
            start_cell = world_to_pixel(
                route_wps[i],
                self.get_parameter("resolution").value,
                self.get_parameter("x_offset").value,
                self.get_parameter("y_offset").value,
                self._cos_rot,
                self._sin_rot,
            )
            end_cell = world_to_pixel(
                route_wps[i + 1],
                self.get_parameter("resolution").value,
                self.get_parameter("x_offset").value,
                self.get_parameter("y_offset").value,
                self._cos_rot,
                self._sin_rot,
            )
            route_cells.extend(waypoints_to_cells(start_cell[0], start_cell[1], end_cell[0], end_cell[1]))
        return route_cells

    def _new_goal(self, msg: PoseStamped) -> None:
        self._logger.info(f"New goal received: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        try:
            robot_pose = self._tf_buffer.lookup_transform(
                self._route_frame_id,
                self._robot_frame_id,
                rclpy.time.Time(),
            )
        except TransformException as e:
            self._logger.error(f"Failed to get robot pose: {e}")
            return

        robot_position = (
            robot_pose.transform.translation.x,
            robot_pose.transform.translation.y,
            robot_pose.transform.translation.z,
        )
        goal_position = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )

        route = self._get_route(robot_position, goal_position)
        if route is not None:
            self._logger.info(f"Publishing found route to {self.get_parameter('route_topic').value}")
            self._pub_route.publish(self._route_to_path_msg(route))

    def _draw_and_visualize_swagger(self):
        swagger_viz = MarkerArray()
        node_viz = Marker()
        node_viz.header.frame_id = self._route_frame_id
        node_viz.ns = "swagger_nodes"
        node_viz.action = Marker.ADD
        node_viz.type = Marker.POINTS
        node_viz.pose.orientation.w = 1.0
        node_viz.scale.x = 0.08
        node_viz.scale.y = 0.08
        node_viz.color.r = 1.0
        node_viz.color.a = 0.4
        node_viz.id = 1000
        for _, node_data in self._swagger.graph.nodes(data=True):
            point = Point()
            point.x = node_data["world"][0]
            point.y = node_data["world"][1]
            node_viz.points.append(point)
        swagger_viz.markers.append(node_viz)

        edge_viz = Marker()
        edge_viz.header.frame_id = self._route_frame_id
        edge_viz.ns = "swagger_edges"
        edge_viz.action = Marker.ADD
        edge_viz.type = Marker.LINE_LIST
        edge_viz.pose.orientation.w = 1.0
        edge_viz.scale.x = 0.02
        edge_viz.color.r = 1.0
        edge_viz.color.b = 1.0
        edge_viz.color.a = 0.8
        edge_viz.id = 1000

        for edge in self._swagger.graph.edges:
            node_1 = self._swagger.graph.nodes[edge[0]]
            node_2 = self._swagger.graph.nodes[edge[1]]
            point_1 = Point()
            point_1.x = node_1["world"][0]
            point_1.y = node_1["world"][1]

            point_2 = Point()
            point_2.x = node_2["world"][0]
            point_2.y = node_2["world"][1]

            edge_viz.points.append(point_1)
            edge_viz.points.append(point_2)

        swagger_viz.markers.append(edge_viz)

        self._logger.info(f"Publishing SWAGGER visualization to {self.get_parameter('swagger_viz_topic').value}")
        self._pub_swagger_viz.publish(swagger_viz)


def main():
    rclpy.init()
    route_planner_node = RoutePlanner("swagger_route_planner")

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(route_planner_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        route_planner_node.destroy_node()
    finally:
        executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
