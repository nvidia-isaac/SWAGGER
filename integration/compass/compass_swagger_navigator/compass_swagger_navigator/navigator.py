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

import enum

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig
from swagger.models import Point as SWAGGERPoint


class ModelType(enum.Enum):
    """Policy model type."""

    ONNX = "onnx"
    JIT = "jit"
    NULL = "null"


SUPPORTED_MODEL_EXTENSIONS = {
    ".onnx": ModelType.ONNX,
    ".jit": ModelType.JIT,
    ".pt": ModelType.JIT,
}


def upsample_points(
    start: list[float] | tuple[float, float],
    goal: list[float] | tuple[float, float],
    max_segment_length: float,
) -> list[tuple[float, float]]:
    """
    Borrowed from: https://github.com/NVlabs/X-MOBILITY/blob/main/ros2_deployment/
    x_mobility_navigator/x_mobility_navigator/x_mobility_navigator.py#L48
    """
    x1, y1 = start
    x2, y2 = goal

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Handle the case where the start and goal are too close (or identical)
    if distance <= max_segment_length:
        return [start, goal]

    # Determine the number of segments based on the maximum segment length
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # Generate the interpolated points
    return [(x1 + (i / num_segments) * (x2 - x1), y1 + (i / num_segments) * (y2 - y1)) for i in range(num_segments + 1)]


def infer_runtime_type(model_path: str) -> ModelType:
    """Infer the runtime type of the model based on the file extension."""
    return next(
        (model_type for ext, model_type in SUPPORTED_MODEL_EXTENSIONS.items() if model_path.endswith(ext)),
        ModelType.NULL,
    )


class CompassSWAGGERNavigator(Node):
    """Compass SWAGGER Navigator node

    This node computes navigation commands using a learned mobility policy.

    High level flow:
    - Inputs:
        * Camera images - front view
        * Odometry - to get the robot's speed
        * Goal poses - the interface for setting targets
        * (Optional) route -- not wired currently.
    - Outputs:
        * Navigation commands.
    """

    def __init__(self, node_name, *, namespace=None, enable_rosout=True):
        super().__init__(node_name, namespace=namespace, enable_rosout=enable_rosout)
        self.declare_parameters(
            namespace="",
            parameters=[
                ("policy_path", rclpy.Parameter.Type.STRING),
                ("occupancy_map_file", rclpy.Parameter.Type.STRING),
                ("nav_command_topic", "cmd_vel"),
                ("image_topic", "front_camera"),
                ("odom_topic", "odom"),
                ("route_topic", "route"),
                ("goal_topic", "goal_pose"),
                ("robot_frame", "torso_base"),
                ("frequency", 10.0),
                ("goal_tolerance", 1.0),
                ("mapless_mode", False),
                # COMPASS parameters
                ("image_width", 960),
                ("image_height", 640),
                ("max_linear_speed_x", 0.8),
                ("max_linear_speed_y", 0.5),
                ("max_angular_speed", 1.0),
                ("num_route_points", 11),
                ("route_vector_size", 4),
                # SWAGGER parameters
                ("resolution", 0.05),
                ("safety_distance", 0.3),
                ("occupancy_threshold", 127),
                ("x_offset", 0.0),
                ("y_offset", 0.0),
                ("rotation", 0.0),
                ("swagger_viz_topic", "swagger_viz"),
            ],
        )

        self._pub_nav_cmd = self.create_publisher(Twist, self._get_param("nav_command_topic"), 10)
        self._pub_path_viz = self.create_publisher(Path, "compass/route_path", 10)
        self._pub_swagger_viz = self.create_publisher(MarkerArray, self.get_parameter("swagger_viz_topic").value, 10)

        self.create_subscription(Image, self._get_param("image_topic"), self._new_image, 10)
        self.create_subscription(Odometry, self._get_param("odom_topic"), self._new_odom, 10)
        self.create_subscription(PoseStamped, self._get_param("goal_topic"), self._new_goal, 10)

        self._timer = self.create_timer(1.0 / self.get_parameter("frequency").value, self._tick)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._logger = self.get_logger()
        self._logger.info(f"Initialized {self.get_name()} with parameters as:")
        for param in self.get_parameters_by_prefix(""):
            self._logger.info(f"\t{param}: {self.get_parameter(param).value}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cv_bridge = CvBridge()
        self._initialize()

    def _initialize(self):
        self._logger.debug("Initializing navigator")

        self._mapless_mode = self._get_param("mapless_mode")
        self._robot_frame = self._get_param("robot_frame")
        self._max_linear_speed_x: float = self._get_param("max_linear_speed_x")
        self._max_linear_speed_y: float = self._get_param("max_linear_speed_y")
        self._max_angular_speed: float = self._get_param("max_angular_speed")
        self._image_width: int = self._get_param("image_width")
        self._image_height: int = self._get_param("image_height")
        self._num_route_points: int = self._get_param("num_route_points")
        self._route_vector_size: int = self._get_param("route_vector_size")
        self._node_freq = self.get_parameter("frequency").get_parameter_value().double_value
        self._goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self._occupancy_map_file = self.get_parameter("occupancy_map_file").get_parameter_value().string_value

        self._runtime_context = None
        self._policy = None
        policy_path = self.get_parameter("policy_path").get_parameter_value().string_value
        self._model_type = infer_runtime_type(policy_path)
        self._load_policy_model(policy_path)

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
        self._draw_and_visualize_swagger()

        self._reset()

    def _reset(self):
        self._last_tick_time = self.get_clock().now()
        self._tf_buffer.clear()
        self._position_2d = np.zeros(3, dtype=np.float32)  # (x, y, angular_z)
        self._camera_image = None
        self._goal = None
        self._ego_speed = np.zeros(1, dtype=np.float32)
        self._action = np.zeros(6, dtype=np.float32)
        self._history = np.zeros((1, 1024), dtype=np.float32)
        self._sample = np.zeros((1, 512), dtype=np.float32)
        self._route_vectors = None
        self._prev_transform = None

    def _tick(self):
        now = self.get_clock().now()
        if now < self._last_tick_time:
            self._reset()
        self._last_tick_time = now

        self._check_goal_reached()
        self._build_route_input()
        self.run_inference()

    def _load_policy_model(self, policy_path: str):
        self._logger.info(f"Loading {policy_path}")
        if self._model_type == ModelType.ONNX:
            self._load_onnx_model(policy_path)
        elif self._model_type == ModelType.JIT:
            self._load_jit_model(policy_path)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def _load_onnx_model(self, policy_path: str):
        self._runtime_context = ort.InferenceSession(policy_path)
        if torch.cuda.is_available():
            self._runtime_context.set_providers(["CUDAExecutionProvider"])
            self._logger.info("CUDA is used for runtime!")
        else:
            self._logger.warn("CUDA not available, using CPU. Expect performance drop")

    def _load_jit_model(self, policy_path: str):
        self._policy = torch.jit.load(policy_path)

    def run_inference(self):
        """Run inference."""
        if self._camera_image is None:
            self._logger.info("Waiting for camera image", throttle_duration_sec=0.5)
            return
        if self._route_vectors is None:
            self._logger.info("Waiting for route vectors", throttle_duration_sec=0.5)
            return

        if self._model_type == ModelType.ONNX:
            self._run_onnx_inference()
        elif self._model_type == ModelType.JIT:
            self._run_jit_inference()
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def _run_jit_inference(self):
        """Run inference using the JIT model."""
        if self._policy is None:
            self._logger.warn("Policy is not loaded, re-launch node and specify policy path")
            return
        image = torch.from_numpy(self._camera_image.reshape(1, 1, *self._camera_image.shape)).to(self._device)
        route = torch.from_numpy(self._route_vectors.reshape(1, 1, *self._route_vectors.shape)).to(self._device)

        speed = torch.from_numpy(self._ego_speed.reshape((1, 1, 1))).to(self._device)
        action = torch.from_numpy(self._action.reshape(1, len(self._action))).to(self._device)
        history = torch.from_numpy(self._history).to(self._device)
        sample = torch.from_numpy(self._sample).to(self._device)

        new_actions, new_history, new_sample = self._policy(image, route, speed, action, history, sample)
        self._action = new_actions.squeeze().detach().cpu().numpy()
        self._history = new_history.detach().cpu().numpy()
        self._sample = new_sample.detach().cpu().numpy()

        self.send_out_commands([float(x) for x in self._action])

    def _run_onnx_inference(self):
        """Run inference using the ONNX model."""
        if self._runtime_context is None:
            self._initialize()

        # - image_input: add batch and channel dimensions to the camera image.
        #   The resulting shape becomes [1, 1, C, H, W], where C, H, W are the
        #   original image dimensions.
        image_input = self._camera_image.reshape(1, 1, *self._camera_image.shape)

        # - route_input: similarly, add two dimensions to the route vectors.
        #   The shape becomes [1, 1, D1, D2] based on the original _route_vectors shape.
        route_input = self._route_vectors.reshape(1, 1, *self._route_vectors.shape)

        # - speed_input: reshape the ego speed into a 3D array with shape [1, 1, 1].
        speed_input = np.array(self._ego_speed).reshape((1, 1, 1))

        # - action_input: add a dimension to the action array; resulting shape is
        #   [1, len(self._action)].
        action_input = self._action.reshape(1, len(self._action))

        output_names = ["action_output", "history_output", "sample_output"]
        outputs = self._runtime_context.run(
            output_names,
            {
                "image": image_input,
                "route": route_input,
                "speed": speed_input,
                "action_input": action_input,
                "history_input": self._history,
                "sample_input": self._sample,
            },
        )

        self._action = outputs[0][0][:]
        self._logger.info(f"Action: {self._action[0]}, {self._action[5]}", throttle_duration_sec=10)

        self._history = outputs[1]
        self._sample = outputs[2]

        self.send_out_commands([float(x) for x in self._action])

    def _new_image(self, image_msg: Image):
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            resized_image = cv2.resize(cv_image, (self._image_width, self._image_height))
            resized_image_msg = self._cv_bridge.cv2_to_imgmsg(resized_image, encoding="bgr8")

            image_channels = int(resized_image_msg.step / resized_image_msg.width)
            image = np.array(resized_image_msg.data).reshape(
                (resized_image_msg.height, resized_image_msg.width, image_channels)
            )
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            self._camera_image = np.ascontiguousarray(image)

        except CvBridgeError as cvb_e:
            self._logger.error(f"Failed to convert image from msg to CV, error: {cvb_e}")

    def _new_odom(self, msg: Odometry):
        self._ego_speed[:] = msg.twist.twist.linear.x

        self._position_2d[0] = msg.pose.pose.position.x
        self._position_2d[1] = msg.pose.pose.position.y
        self._position_2d[2] = msg.pose.pose.orientation.z

    def _new_goal(self, goal_msg: PoseStamped):
        self._logger.info(
            f"Received new goal: [{goal_msg.pose.position.x}, {goal_msg.pose.position.y}], "
            f"in frame: {goal_msg.header.frame_id}"
        )
        self._goal = goal_msg

        # Reset history and sample
        self._history = np.zeros((1, 1024), dtype=np.float32)
        self._sample = np.zeros((1, 512), dtype=np.float32)
        self._action = np.zeros(6, dtype=np.float32)
        self._route_vectors = None

    def _build_route_input(self):
        if self._goal is None:
            self._logger.info("Waiting for goal", throttle_duration_sec=2.0)
            return
        if self._mapless_mode:
            self._compose_mapless_route()
        else:
            self._generate_swagger_route()

    def _generate_swagger_route(self) -> None:
        self._logger.debug("Generating swagger route")
        try:
            robot_to_goal_transform = self._tf_buffer.lookup_transform(
                self._goal.header.frame_id, self._robot_frame, Time()
            )
        except TransformException as ex:
            self._logger.error(f"Transform Error: {self._robot_frame} to {self._goal.header.frame_id}: {ex}")
            return

        start_point = SWAGGERPoint(
            x=robot_to_goal_transform.transform.translation.x, y=robot_to_goal_transform.transform.translation.y
        )
        goal_point = SWAGGERPoint(x=self._goal.pose.position.x, y=self._goal.pose.position.y)
        route = self._swagger.find_route(start_point, goal_point)
        if not route:
            self._logger.error(f"Failed to get route for start: {start_point} and goal: {goal_point}")
            return

        try:
            route_to_robot_transform = self._tf_buffer.lookup_transform(self._robot_frame, "map", Time())
        except TransformException as ex:
            self._logger.error(f"Transform Error {self._robot_frame} to map: {ex}")
            return

        header_msg = Header()
        header_msg.stamp = self.get_clock().now().to_msg()
        header_msg.frame_id = "map"
        route_poses = []
        for point in route:
            pose = PoseStamped()
            pose.header = header_msg
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.w = 1.0  # Default orientation
            transformed_pose = do_transform_pose(pose.pose, route_to_robot_transform)
            route_poses.append((transformed_pose.position.x, transformed_pose.position.y))

        self._prepare_route_vectors(route_poses)

    def _compose_mapless_route(self):
        self._logger.debug("Composing mapless route")
        if self._goal is None:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._robot_frame,
                self._goal.header.frame_id,
                Time(),
            )
        except TransformException as ex:
            self._logger.warn(f"Could not transform {self._robot_frame} to {self._goal.header.frame_id}: {ex}")
            return

        self._logger.debug("Computing a new mapless route")

        goal_in_robot_frame = do_transform_pose(self._goal.pose, transform)
        route_poses = upsample_points(
            [0.0, 0.0],
            [goal_in_robot_frame.position.x, goal_in_robot_frame.position.y],
            1.0,
        )
        self._prepare_route_vectors(route_poses)

    def _prepare_route_vectors(self, route_poses: list[tuple[float, float]]) -> np.ndarray:
        num_poses = min(len(route_poses), self._num_route_points)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first self._num_route_points and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (self._num_route_points - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx])
        self._route_vectors = np.zeros((self._num_route_points - 1, self._route_vector_size), np.float32)
        for idx in range(self._num_route_points - 1):
            self._route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx], selected_route_positions[idx + 1]),
                axis=0,
            )
        self._visualize_route(selected_route_positions, self._robot_frame)

    def send_out_commands(self, nav_commands: list[float]):
        """Send out twist commands to the robot.

        Args:
            nav_commands: A numpy array of shape (6,) of
                the linear[x, y, z] and angular[x, y, z] velocities.
        """
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = np.clip(nav_commands[0], -self._max_linear_speed_x, self._max_linear_speed_x)
        cmd_vel_msg.linear.y = np.clip(nav_commands[1], -self._max_linear_speed_y, self._max_linear_speed_y)
        cmd_vel_msg.angular.z = np.clip(nav_commands[5], -self._max_angular_speed, self._max_angular_speed)
        self._pub_nav_cmd.publish(cmd_vel_msg)

    def _get_param(self, name: str):
        return self.get_parameter(name).value

    def _check_goal_reached(self) -> bool:
        if self._goal is None:
            return
        goal_2d = np.array([self._goal.pose.position.x, self._goal.pose.position.y, self._goal.pose.orientation.z])
        if np.linalg.norm(self._position_2d - goal_2d) < self._goal_tolerance:
            self._goal = None
            self._route_vectors = None
            self.send_out_commands([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Stop the robot

    def _visualize_route(self, selected_route_positions: list[list[float]], frame_id: str) -> None:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        for point in selected_route_positions:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self._pub_path_viz.publish(path_msg)

    def _draw_and_visualize_swagger(self):
        swagger_viz = MarkerArray()
        node_viz = Marker()
        node_viz.header.frame_id = "map"
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
        edge_viz.header.frame_id = "map"
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
        self._pub_swagger_viz.publish(swagger_viz)


def main():
    rclpy.init()
    navigator_node = CompassSWAGGERNavigator("compass_swagger_navigator")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(navigator_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        navigator_node.get_logger().info("Keyboard interrupt, shutting down.")
        navigator_node.destroy_node()
    finally:
        executor.remove_node(navigator_node)
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
