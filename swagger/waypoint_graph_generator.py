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

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cupy as cp
import cv2
import networkx as nx
import numba
import numpy as np
import skan
from cucim.skimage.morphology import thin
from rtree import index
from scipy.spatial import Delaunay, cKDTree

from swagger.logger import Logger
from swagger.models import Point
from swagger.utils import pixel_to_world, world_to_pixel


@dataclass
class Color:
    r: int  # Red value
    g: int  # Green value
    b: int  # Blue value

    def __post_init__(self):
        for color, name in [(self.r, "Red"), (self.g, "Green"), (self.b, "Blue")]:
            if not (0 <= color <= 255):
                raise ValueError(f"{name} value must be between 0 and 255")

    def to_tuple(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b


@dataclass
class WaypointGraphGeneratorConfig:
    """Configuration for waypoint graph generation algorithm.

    All distance parameters are specified in meters and are converted
    to pixel units internally based on the resolution when needed.
    """

    # Graph generation parameters (in meters)
    skeleton_sample_distance: float = 1.5  # Distance between samples along skeleton image
    boundary_inflation_factor: float = 1.5  # Factor to inflate boundaries by safety distance (unitless)
    boundary_sample_distance: float = 2.5  # Distance between samples along contour
    free_space_sampling_threshold: float = 1.5  # Maximum distance from obstacles

    # Graph pruning parameters (in meters)
    merge_node_distance: float = 0.25  # Maximum distance to merge nodes
    min_subgraph_length: float = 0.25  # Minimum total edge length to keep

    # Colors for visualization
    edge_color: Color = field(default_factory=lambda: Color(r=0, g=0, b=255))
    node_color: Color = field(default_factory=lambda: Color(r=255, g=0, b=0))

    # Function flags
    use_skeleton_graph: bool = True
    use_boundary_sampling: bool = True
    use_free_space_sampling: bool = True
    use_delaunay_shortcuts: bool = True
    prune_graph: bool = True

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # List of float parameters that must be positive
        distance_params = [
            "skeleton_sample_distance",
            "boundary_sample_distance",
            "free_space_sampling_threshold",
            "merge_node_distance",
            "min_subgraph_length",
        ]

        # Check each parameter
        for name in distance_params:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"Parameter '{name}' must be greater than 0, got {value}")

        if self.boundary_inflation_factor <= 1.0:
            raise ValueError("Parameter 'boundary_inflation_factor' must be greater than 1.0")


class WaypointGraphGenerator:
    def __init__(
        self, config: WaypointGraphGeneratorConfig = WaypointGraphGeneratorConfig(), logger_level: int = logging.INFO
    ):
        self._graph: nx.Graph | None = None  # Store the current graph
        self._original_map: np.ndarray | None = None  # Store original map for visualization
        # Store inflated map for collision checks, shape is the same as the original map
        self._inflated_map: np.ndarray | None = None
        self._config = config  # Store graph generation configuration
        self._resolution: float | None = None  # Store resolution for coordinate conversion
        self._safety_distance: float | None = None  # Store robot radius for visualization
        self._occupancy_threshold: int = 127  # Store occupancy threshold
        # Store transform parameters
        self._x_offset: float = 0.0
        self._y_offset: float = 0.0
        self._rotation: float = 0.0  # Rotation in radians
        self._cos_rot: float = 1.0
        self._sin_rot: float = 0.0
        # Initialize logger
        self._logger = Logger(__name__, level=logger_level)

        self._node_map: np.ndarray | None = (
            None  # Map of nodes for quick lookup, pixel values are node indices if not NaN
        )

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def build_graph_from_grid_map(
        self,
        image: np.ndarray,
        resolution: float,
        safety_distance: float,
        occupancy_threshold: int = 127,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        rotation: float = 0.0,
    ) -> nx.Graph:
        """
        Build a waypoint graph from an occupancy grid map.

        Args:
            image: Occupancy grid (0-255, where values <= threshold are considered occupied)
            resolution: Meters per pixel in the map
            safety_distance: Radius of the robot in meters
            occupancy_threshold: Threshold to determine occupied cells (0-255)
            x_offset: Translation in the x direction (meters)
            y_offset: Translation in the y direction (meters)
            rotation: Rotation about the Z axis (radians)

        Returns:
            NetworkX.Graph object containing the waypoint graph
        """
        self._logger.info("Building graph from grid map...")
        self._resolution = resolution
        self._safety_distance = safety_distance
        self._original_map = copy.deepcopy(image)
        self._occupancy_threshold = occupancy_threshold
        self._x_offset = x_offset

        # When using ROS coordinates, we need to add the image height to the y_offset
        # This shifts the origin from top-left to bottom-left
        image_height = image.shape[0]
        self._y_offset = y_offset + (image_height * resolution)  # Add image height in meters

        self._cos_rot = np.cos(rotation)
        self._sin_rot = np.sin(rotation)
        self._node_map = None

        # Check if map is completely free before performing distance transform
        free_map = (self._original_map > self._occupancy_threshold).astype(np.uint8)
        if np.all(free_map):
            # Map is completely free (all values > threshold), create grid graph directly
            self._graph = self._create_grid_graph(
                image.shape, grid_sample_distance=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )
            self._build_nearest_node_map(self._graph)
            return self._graph

        # Map has obstacles, perform distance transform
        self._distance_transform(free_map)

        # Build graph from the skeleton of the map
        if self._config.use_skeleton_graph:
            graph = self._build_graph_from_skeleton(
                skeleton_sample_distance=self._to_pixels_int(self._config.skeleton_sample_distance)
            )
        else:
            graph = nx.Graph()

        # Sample nodes along obstacle boundaries
        if self._config.use_boundary_sampling:
            self._sample_obstacle_boundaries(
                graph, sample_distance=self._to_pixels_int(self._config.boundary_sample_distance)
            )

        # Sample nodes in free space
        if self._config.use_free_space_sampling:
            self._sample_free_space(
                graph, distance_threshold=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )

        # Add shortcuts
        if self._config.use_delaunay_shortcuts:
            self._add_delaunay_shortcuts(graph)

        # Prune unnecessary nodes and edges from the graph
        if self._config.prune_graph:
            self._prune_graph(graph)

        # Add world coordinates to nodes
        graph = self._to_world_coordinates(graph)

        # Store and convert the graph to CSR format
        self._graph = graph
        self._logger.info(f"Final graph has {len(graph.nodes)} nodes and {len(graph.edges)} unique edges")
        self._build_nearest_node_map(self._graph)
        return self._graph

    def _astar_heuristic(self, n1, n2):
        """A* heuristic for the graph."""
        world_n1 = self._graph.nodes[n1]["world"]
        world_n2 = self._graph.nodes[n2]["world"]
        return np.linalg.norm([world_n1[0] - world_n2[0], world_n1[1] - world_n2[1]])

    def find_route(self, start: Point, goal: Point, shortcut_distance: float = 0.0) -> list[Point]:
        """Find a route between two points in the graph. If the distance between the start and goal is less than
        shortcut_distance, we check if the line between them is collision free and return the start and goal points
        if it is. Otherwise, we use A* to find a route on the graph.
        """

        distance = np.linalg.norm([start.x - goal.x, start.y - goal.y])
        if distance < shortcut_distance:
            start_pixel = self._world_to_pixel(start)
            goal_pixel = self._world_to_pixel(goal)
            if not self._check_line_collision(start_pixel, goal_pixel):
                return [start, goal]

        try:
            start_node, goal_node = self.get_node_ids([start, goal])
        except Exception as e:
            self._logger.error(f"Error getting start and goal nodes: {start} - {goal}: {e}")
            return []

        if start_node is None:
            self._logger.error(f"{start} Start is out of bounds")
            return []
        if goal_node is None:
            self._logger.error(f"{goal} Goal is out of bounds")
            return []

        try:
            route_nodes = nx.astar_path(self._graph, start_node, goal_node, heuristic=self._astar_heuristic)
            return (
                [start]
                + [
                    Point(
                        x=self._graph.nodes[node]["world"][0],
                        y=self._graph.nodes[node]["world"][1],
                        z=self._graph.nodes[node]["world"][2],
                    )
                    for node in route_nodes
                ]
                + [goal]
            )
        except nx.NetworkXNoPath:
            self._logger.error(f"No path found for start: {start} and goal: {goal}")
            return []

    def _distance_transform(self, free_map: np.ndarray):
        """Inflate obstacles in the occupancy grid using a distance transform."""
        # Pad the binary map by 1 pixel on all sides to avoid nodes being created on the edges of the map
        free_map = np.pad(free_map, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        # Compute the distance transform
        self._dist_transform = cv2.distanceTransform(free_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Filter the distance transform by the robot's radius
        self._inflated_map = (self._dist_transform < self._safety_distance / self._resolution).astype(np.uint8)
        # Unpad the distance transform to get the original map shape
        self._inflated_map = self._inflated_map[1:-1, 1:-1]

    def _create_grid_graph(self, shape: tuple[int, int], grid_sample_distance: int) -> nx.Graph:
        """Create a grid graph for completely free maps with a margin from the borders."""
        self._logger.info("Creating grid for completely free map with margin...")
        height, width = shape

        # Calculate margin based on robot radius and resolution
        margin = int(self._safety_distance / self._resolution)

        # For very small maps, just create a single node
        if height <= 2 * margin or width <= 2 * margin:
            raise ValueError(
                f"Map is too small to create a grid graph (height: {height}, width: {width}, safety_distance:"
                f" {self._safety_distance}, resolution: {self._resolution})"
            )

        # Ensure minimum step size of 1 pixel
        # Choose step size for grid:
        # - Minimum of 1 pixel to ensure we can always create a grid
        # - Maximum of `free_space_sampling_threshold` pixels to avoid too sparse sampling in large maps
        # - For medium maps, use half the smallest map dimension (after margins)
        #   to create a reasonable number of waypoints
        step = max(1, min(grid_sample_distance, min(height - 2 * margin, width - 2 * margin) // 2))
        graph = nx.Graph()

        # Create grid nodes starting from corners with a margin
        for y in range(margin, height - margin, step):
            for x in range(margin, width - margin, step):
                graph.add_node((y, x))
                # Connect to right neighbor
                if x + step < width - margin:
                    graph.add_edge((y, x), (y, x + step), weight=step, edge_type="grid")
                # Connect to bottom neighbor
                if y + step < height - margin:
                    graph.add_edge((y, x), (y + step, x), weight=step, edge_type="grid")
                # Connect to diagonal neighbor
                if x + step < width - margin and y + step < height - margin:
                    graph.add_edge((y, x), (y + step, x + step), weight=step * np.sqrt(2), edge_type="grid")

        self._logger.info(f"Created grid graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return self._to_world_coordinates(graph)

    def _to_world_coordinates(self, graph: nx.Graph) -> nx.Graph:
        """Convert nodes from pixel coordinates to world coordinates.

        Takes a NetworkX graph with nodes in pixel coordinates (y,x) and converts them to world coordinates (x,y,z)
        using the current transform parameters (resolution, offset, rotation). Edge weights are scaled by resolution
        to convert from pixels to meters.

        Args:
            graph: NetworkX graph with nodes in pixel coordinates (y,x)

        Returns:
            NetworkX graph with nodes converted to world coordinates (x,y,z) tuples, preserving all attributes.
            Original pixel coordinates are stored in node attribute 'pixel'.
        """
        world_graph = nx.Graph()
        pixel_to_id_lookup = {}
        # Convert each node to world coordinates and store mapping
        for i, (node, data) in enumerate(graph.nodes(data=True)):
            row, col = node
            world_point = self._pixel_to_world(row, col)
            world_point_array = tuple((world_point.x, world_point.y, world_point.z))
            pixel = (int(row), int(col))
            world_graph.add_node(i, world=world_point_array, pixel=pixel, **data)
            pixel_to_id_lookup[node] = i

        # Convert edge weights from pixels to meters and copy edges
        for src, dst, data in graph.edges(data=True):
            data["weight"] = float(data["weight"]) * self._resolution
            world_graph.add_edge(pixel_to_id_lookup[src], pixel_to_id_lookup[dst], **data)

        return world_graph

    def _build_graph_from_skeleton(self, skeleton_sample_distance: int) -> np.ndarray:
        """Generate a graph from the skeleton of the inflated map."""
        self._logger.info("Generating skeleton...")
        try:
            skeleton_image = thin((1 - self._inflated_map).view(cp.uint8))
        except RuntimeError as e:
            from skimage.morphology import skeletonize

            self._logger.warning(f"Failed to generate skeleton with cucim: {e}. Falling back to skimage.")
            skeleton_image = skeletonize(1 - self._inflated_map)

        if skeleton_image.sum() == 0:
            self._logger.warning("No skeleton found in map")
            return nx.Graph()

        self._logger.info("Building graph from skeleton...")
        # Convert to numpy if it's a cupy array, otherwise use as-is
        if hasattr(skeleton_image, "get"):  # cupy array
            skeleton_image_np = skeleton_image.get()
        else:  # numpy array
            skeleton_image_np = skeleton_image
        skeleton = skan.Skeleton(skeleton_image_np)
        graph = nx.Graph()

        for i in range(skeleton.n_paths):
            branch = skeleton.path_coordinates(i)
            # Count the number of segments in the branch
            num_segments = max(1, len(branch) // skeleton_sample_distance)
            segment_length = len(branch) // num_segments
            for j in range(num_segments):
                start = branch[j * segment_length]
                end = branch[min((j + 1) * segment_length, len(branch) - 1)]
                if self._check_line_collision(start, end):
                    continue
                dist = np.linalg.norm(end - start)
                graph.add_edge((start[0], start[1]), (end[0], end[1]), weight=dist, edge_type="skeleton")
        self._logger.info(f"Created skeleton graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph

    def _check_line_collision(self, p0, p1):
        """
        Check if the line between two points intersects with any obstacles.

        Args:
            p0: Coordinates of the start point (y, x)
            p1: Coordinates of the end point (y, x)

        Returns:
            True if there is a collision, False otherwise
        """
        # Bresenham's line algorithm to get the pixels the line passes through
        y0, x0 = p0
        y1, x1 = p1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self._inflated_map[y0, x0]:
                return True  # Return early if a collision is detected
            if (y0 == y1) and (x0 == x1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return False  # No collision detected

    def _sample_free_space(self, graph, distance_threshold: float):
        """
        Iteratively sample free space in a map by identifying and adding nodes at local maxima
        of large distance areas until no such areas remain.

        Args:
            graph: The graph to which new nodes will be added.
            distance_threshold: The threshold to identify areas with large distances.
        """
        # Initialize distance map with a large value
        distance_map = np.full(self._original_map.shape, np.inf, dtype=np.float64)

        # Set occupied pixels to 0
        distance_map[self._original_map <= self._occupancy_threshold] = 0

        # Set node pixels to 0
        for node in graph.nodes():
            row, col = node
            distance_map[row, col] = 0

        # Initialize R-tree index
        idx = index.Index()

        # Initialize kernel for dilation
        kernel = np.ones((3, 3), np.uint8)

        while True:
            # Use distance transform to calculate distances to the nearest node
            distance_map = cv2.distanceTransform(
                (distance_map > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )

            # Identify large distance areas
            large_distance_areas = (distance_map > distance_threshold).astype(np.uint8)

            if not np.any(large_distance_areas):
                break

            # Dilate the large distance areas
            dilated = cv2.dilate(distance_map, kernel)

            # Find local maxima
            local_maxima_mask = (distance_map == dilated) & (large_distance_areas > 0)
            local_maxima_coords = np.column_stack(np.where(local_maxima_mask))

            # Add local maxima as nodes in the graph
            for coord in local_maxima_coords:
                row, col = coord
                # Check if there are any nodes within distance_threshold/2
                half_threshold = distance_threshold / 2
                bounding_box = (col - half_threshold, row - half_threshold, col + half_threshold, row + half_threshold)
                intersections = list(idx.intersection(bounding_box))
                if len(intersections) == 0:
                    graph.add_node((row, col))
                    idx.insert(len(graph.nodes) - 1, (col, row, col, row))  # Insert node into R-tree
                    distance_map[row, col] = 0

    def _add_delaunay_shortcuts(self, graph: nx.Graph):
        """
        Create shortcuts based on Delaunay triangulation of nodes.

        Args:
            graph: NetworkX graph to add shortcuts to
        """
        self._logger.info("Adding Delaunay shortcuts...")
        nodes = list(graph.nodes())
        if len(nodes) < 3:  # Need at least 3 points for triangulation
            return

        # Convert nodes to numpy array and compute triangulation
        node_coords = np.array(nodes)
        try:
            tri = Delaunay(node_coords)
        except Exception as e:
            self._logger.error(f"Skipping Delaunay triangulation: {e}")
            return

        # Collect edges for batch processing
        edge_candidates = set()
        for simplex in tri.simplices:
            for i in range(3):
                n1, n2 = tuple(sorted([nodes[simplex[i]], nodes[simplex[(i + 1) % 3]]]))
                if not graph.has_edge(n1, n2):
                    edge_candidates.add((n1, n2))

        if not edge_candidates:
            return
        # Add valid edges
        for n1, n2 in edge_candidates:
            if not self._check_line_collision(n1, n2):
                dist = np.sqrt((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2)
                graph.add_edge(n1, n2, weight=dist, edge_type="delaunay")

    def _sample_obstacle_boundaries(self, graph: nx.Graph, sample_distance: float = 50) -> list[tuple[int, int]]:
        """
        Sample nodes along obstacle boundaries.

        Args:
            graph: Existing graph
            sample_distance: Distance between samples along contour (in pixels).
                           Defaults to 50 (pixels).
        """
        # Find contours of inflated obstacles
        contours = self._find_obstacle_contours(
            self._config.boundary_inflation_factor * self._safety_distance / self._resolution
        )

        initial_num_nodes = len(graph.nodes())

        # Process each contour
        for contour in contours:
            contour_nodes = []

            # Process each vertex in the contour
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]  # Wrap around to first point

                # Convert first point to y,x format
                row_1, col_1 = int(p1[1]), int(p1[0])
                contour_nodes.append((row_1, col_1))
                graph.add_node((row_1, col_1), node_type="boundary")

                # Calculate distance to next vertex
                segment_length = np.linalg.norm(p2 - p1)
                # Add intermediate points
                num_intermediate = int(segment_length / sample_distance)
                # Skip first and last points
                intermediate_points = np.linspace(p1, p2, num=num_intermediate, endpoint=False).astype(int).tolist()[1:]
                for point in intermediate_points:
                    # Interpolate point
                    col, row = point

                    # Add interpolated point
                    contour_nodes.append((row, col))
                    graph.add_node((row, col), node_type="boundary")

            # Connect consecutive nodes along the contour
            self._connect_contour_nodes(contour_nodes, graph)

        num_nodes_added = len(graph.nodes()) - initial_num_nodes
        self._logger.info(f"Added {num_nodes_added} nodes along obstacle boundaries")

    def _find_obstacle_contours(self, boundary_inflation: float) -> list[np.ndarray]:
        """Find contours of inflated obstacles using the distance map."""
        # Use the distance transform to filter obstacles
        filtered_obstacles = (self._dist_transform >= boundary_inflation).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(filtered_obstacles, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        return contours

    def _is_within_bounds(self, row: int, col: int) -> bool:
        """Check if a point is within the bounds of the map."""
        return 0 <= row < self._inflated_map.shape[0] and 0 <= col < self._inflated_map.shape[1]

    def _is_valid_point(self, row: int, col: int) -> bool:
        """Check if a point is within bounds and in free space."""
        return self._is_within_bounds(row, col) and not self._inflated_map[row, col]

    def _connect_contour_nodes(self, contour_nodes: list[tuple[int, int]], graph: nx.Graph):
        """Connect consecutive nodes along the contour."""
        for i in range(len(contour_nodes)):
            n1 = contour_nodes[i]
            n2 = contour_nodes[(i + 1) % len(contour_nodes)]

            if not self._check_line_collision(n1, n2):
                dist = np.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
                graph.add_edge(n1, n2, weight=dist, edge_type="contour")

    def _prune_graph(self, graph: nx.Graph):
        """Remove isolated nodes, small subgraphs, and merge close nodes."""
        self._logger.info("Pruning graph...")

        if len(graph.nodes()) > 0:
            # Merge nodes that are close to each other
            self._merge_close_nodes(graph, threshold=self._to_pixels_int(self._config.merge_node_distance))

        # Remove isolated nodes and small subgraphs
        graph.remove_nodes_from(list(nx.isolates(graph)))

        # Find connected components and filter small ones
        components = list(nx.connected_components(graph))
        for component in components:
            subgraph = graph.subgraph(component)
            total_length = sum(d["weight"] for _, _, d in subgraph.edges(data=True))
            if total_length < self._to_pixels_int(self._config.min_subgraph_length):
                graph.remove_nodes_from(component)

    def _merge_close_nodes(self, graph: nx.Graph, threshold: float):
        """Merge nodes that are within a certain distance threshold."""
        while True:
            nodes = list(graph.nodes())
            node_coords = np.array(nodes)
            tree = cKDTree(node_coords)

            # Find pairs of nodes that are close to each other
            pairs = tree.query_pairs(r=threshold)

            if not pairs:
                break  # Exit loop if no pairs are found

            merged = False  # Track if any merge happened in this iteration

            for n1_idx, n2_idx in pairs:
                n1 = nodes[n1_idx]
                n2 = nodes[n2_idx]

                # Skip if n2 has already been removed
                if not graph.has_node(n2):
                    continue

                # Merge n2 into n1 and update edges
                # Check if all n2's neighbors can connect to n1 without collision
                neighbors = [n for n in graph.neighbors(n2) if n != n1]
                if neighbors:
                    collision = False
                    for dst in neighbors:
                        if self._check_line_collision(n1, dst):
                            collision = True
                            break

                    # Only merge if all connections are valid
                    if not collision:
                        for neighbor in neighbors:
                            dist = np.sqrt((neighbor[0] - n1[0]) ** 2 + (neighbor[1] - n1[1]) ** 2)
                            graph.add_edge(n1, neighbor, weight=dist, edge_type="merge")
                        graph.remove_node(n2)
                        merged = True  # A merge occurred
                else:
                    # No neighbors to check, safe to remove
                    graph.remove_node(n2)
                    merged = True  # A merge occurred

            if not merged:
                break  # Exit loop if no merges occurred in this iteration

    def _pixel_to_world(self, row: float, col: float) -> Point:
        """Convert pixel coordinates to world coordinates with the current transform."""
        return pixel_to_world(row, col, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot)

    def _world_to_pixel(self, point: Point) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates with the inverse transform."""
        return world_to_pixel(point, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot)

    def visualize_graph(
        self,
        output_dir: str = ".",
        output_filename: str = "waypoint_graph.png",
    ) -> None:
        """Create and save a visualization of the graph and original map."""
        if self._graph is None:
            raise RuntimeError("No graph has been built yet")

        self._logger.info("Starting visualization...")

        # Graph visualization using self._original_map
        map_vis = cv2.cvtColor(self._original_map, cv2.COLOR_GRAY2BGR)

        # Draw edges in red
        lines = []
        for src, dst in self._graph.edges():
            src_pixel = self._graph.nodes[src]["pixel"]
            dst_pixel = self._graph.nodes[dst]["pixel"]
            lines.append([[src_pixel[1], src_pixel[0]], [dst_pixel[1], dst_pixel[0]]])
        lines = np.array(lines)
        edge_color = self._config.edge_color.to_tuple()
        cv2.polylines(map_vis, lines, False, edge_color, 1)

        # Draw nodes in blue
        node_radius = 2
        node_color = self._config.node_color.to_tuple()
        for _, pixel in self._graph.nodes(data="pixel"):
            y, x = pixel
            cv2.circle(map_vis, (x, y), node_radius, node_color, -1)

        output_path = os.path.join(output_dir, output_filename)
        if not cv2.imwrite(output_path, map_vis):
            raise RuntimeError(f"Failed to save visualization to {output_path}")

        self._logger.info(f"Saved visualization to {output_path}")

    def get_node_ids(self, points: list[Point]) -> list[Optional[int]]:
        """
        Find the nearest graph node to each query point on the node lookup map.

        Args:
            points: List of Point objects containing x, y, z coordinates in the world frame

        Returns:
            List of node indices (None if no valid path exists)
        """
        if self._graph is None:
            raise RuntimeError("No graph has been built yet")

        if self._node_map is None:
            self._build_nearest_node_map(self._graph)

        results = []
        for point in points:
            y, x = self._world_to_pixel(point)
            if not self._is_within_bounds(y, x):
                results.append(None)
                continue
            label = self._node_map[y, x]
            if label < 0:
                results.append(None)
            else:
                results.append(int(label))

        return results

    @numba.njit
    def _flood_fill_numba(node_map, nodes, width):
        """Flood fill the node map with the node ids."""
        directions = [-width, width, -1, 1]
        queue = []

        for i, pixel in nodes:
            if 0 <= pixel < len(node_map):
                node_map[pixel] = i
                queue.append(pixel)

        for id in queue:
            for dd in directions:
                nid = id + dd
                # Check bounds and handle edge cases
                if 0 <= nid < len(node_map):
                    # Check if we're not wrapping around rows incorrectly
                    current_row = id // width
                    new_row = nid // width
                    if abs(new_row - current_row) <= 1 and node_map[nid] == -1:
                        node_map[nid] = node_map[id]
                        queue.append(nid)

    def _build_nearest_node_map(self, graph: nx.Graph):
        """Build a nearest node map from the graph for quick lookup."""
        self._logger.info("Building nearest node map...")

        height, width = self._original_map.shape
        self._node_map = np.full((height, width), -1, dtype=np.int32)
        self._node_map[self._original_map <= self._occupancy_threshold] = -2

        if len(graph) == 0:  # Empty graph
            return

        graph_pixels = np.array([(i, y * width + x) for i, (y, x) in graph.nodes(data="pixel")], dtype=np.int32)
        node_map = self._node_map.reshape(-1)
        WaypointGraphGenerator._flood_fill_numba(node_map, graph_pixels, width)

        self._logger.info("Nearest node map built")

    def _to_pixels(self, meters: float) -> float:
        """Convert a distance from meters to pixels."""
        if self._resolution is None:
            raise ValueError("Resolution not set. Call build_graph_from_grid_map first.")
        return meters / self._resolution

    def _to_pixels_int(self, meters: float) -> int:
        """Convert a distance from meters to pixels and round to integer."""
        return int(self._to_pixels(meters))
