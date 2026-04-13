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

# Modified --- added math
import math

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
    
    # Debug flag to control debug file output
    debug: bool = False

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
        self._known_points: list[tuple[float, float]] = []

    def _debug_print(self, *args, **kwargs):
        """Print debug messages only if debug flag is enabled."""
        if self._config.debug:
            print(*args, **kwargs)

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
        known_points: list = None, # Modified --- added the concept of known points
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
        self._image_shape = image.shape[:2]

        self._logger.info("Building graph from grid map...")
        self._resolution = resolution
        self._safety_distance = safety_distance
        self._original_map = copy.deepcopy(image)
        self._occupancy_threshold = occupancy_threshold
        self._x_offset = x_offset

        self._debug_print(f"[DEBUG internal] received x_offset={x_offset}, self._resolution={resolution}")
        self._debug_print(f"[DEBUG internal] effective world offset applied? {x_offset / resolution} pixels worth")

        # When using ROS coordinates, we need to add the image height to the y_offset
        # This shifts the origin from top-left to bottom-left
        image_height = image.shape[0]
        #self._y_offset = y_offset + (image_height * resolution)  # Add image height in meters
        self._y_offset = y_offset

        self._cos_rot = np.cos(rotation)
        self._sin_rot = np.sin(rotation)
        self._node_map = None

        # Check if map is completely free before performing distance transform
        free_map = (self._original_map > self._occupancy_threshold).astype(np.uint8)
        self._debug_save_stage("01_free_map", free_map, color=True)

        if np.all(free_map):
            # Map is completely free (all values > threshold), create grid graph directly
            self._graph = self._create_grid_graph(
                image.shape, grid_sample_distance=self._to_pixels_int(self._config.free_space_sampling_threshold)
            )
            self._build_nearest_node_map(self._graph)
            return self._graph

        # Map has obstacles, perform distance transform
        self._distance_transform(free_map)
        self._debug_save_stage("02_inflated_map", self._inflated_map)

        # Modified: mask known points before skeletonization
        #if not hasattr(self, "_visited_mask") or self._visited_mask is None:
            # Initialize the persistent visited-mask once (same size as map)
            #self._visited_mask = np.zeros_like(self._inflated_map, dtype=np.uint8)

        self._visited_mask = np.zeros_like(self._inflated_map, dtype=np.uint8)

        inflated_for_skeleton = self._inflated_map.copy()
        self._debug_save_stage("03a1_inflated_for_skeleton", inflated_for_skeleton)
        known_points_px: list[tuple[int, int]] = []
        visible_known_world: list[tuple[float, float]] = []
        pending_known_nodes: list[tuple[int, int, tuple[float, float, float]]] = []
        if not known_points:
            known_points = list(getattr(self, "_known_points", []))

        self._debug_print(f"[DEBUG mask] known pixels: {len(known_points) if known_points else 0}")

        if known_points:
            for (x, y) in known_points:
                pix = self._world_to_pixel(Point(x=x, y=y, z=0.0))
                if pix is None:
                    continue
                row, col = pix
                if 0 <= row < inflated_for_skeleton.shape[0] and 0 <= col < inflated_for_skeleton.shape[1]:
                    if not self._is_free_pixel(row, col):
                        self._debug_print(f"[DEBUG] Skipping known point {(x, y)} — falls in occupied space.")
                        continue
                    known_points_px.append((row, col))
                    visible_known_world.append((x, y))
                    world = self._pixel_to_world(row, col)
                    world_tuple = self._point_to_tuple(world)
                    pending_known_nodes.append((row, col, world_tuple))
                else:
                    self._debug_print(f"[DEBUG] Skipping known point {(x, y)} — outside current frame bounds.")

            self._debug_print(f"[DEBUG] Converted {len(known_points_px)} known points inside frame.")

        inflated_tmp = inflated_for_skeleton.copy()

        graph = nx.Graph()

        # ------------------------------------------------------------
        # Build graph from the skeleton of the map
        if self._config.use_skeleton_graph:
            skeleton_graph = self._build_graph_from_skeleton(
                skeleton_sample_distance=self._to_pixels_int(self._config.skeleton_sample_distance)
            )
        else:
            graph = nx.Graph()

        for row, col, world_tuple in pending_known_nodes:
            if (row, col) not in graph:
                graph.add_node(
                    (row, col),
                    node_type="known",
                    pixel=(row, col),
                    world=world_tuple,
                    origin="known",
                )

        # Sample nodes along obstacle boundaries
        if self._config.use_boundary_sampling:
            self._sample_obstacle_boundaries(
                graph, sample_distance=self._to_pixels_int(self._config.boundary_sample_distance)
            )

        new_points_px = []
        # Modified --- sampling known points, else sample points in free space
        if self._config.use_free_space_sampling:
            if known_points is not None and len(known_points) > 0:
                # Convert known world points → pixel coordinates
                known_points_px = [
                    self._world_to_pixel(Point(x=x, y=y, z=0.0))
                    for (x, y) in known_points
                ]

                # Filter out any None or malformed points (outside current map)
                known_points_px = [
                    p for p in known_points_px
                    if p is not None and len(p) == 2
                ]

                if len(known_points_px) == 0:
                    self._logger.warning(
                        "[WARN] All known points were out of frame — skipping incremental sampling."
                    )
                    new_points_px = []
                else:
                    new_points_px = self._sample_free_space_incremental(
                        graph,
                        visible_known_world,
                        distance_threshold=self._to_pixels_int(
                            self._config.free_space_sampling_threshold
                        )
                    )
            else:
                self._debug_print(f"[DEBUG] Entered sample_free_space")
                new_points_px = self._sample_free_space(
                    graph,
                    distance_threshold=self._to_pixels_int(
                        self._config.free_space_sampling_threshold
                    ),
                )

        # Remove any stray samples that landed on inflated cells before visualization
        self._remove_invalid_free_nodes(graph)
        new_points_px = [pt for pt in new_points_px if self._is_free_pixel(pt[0], pt[1])]
        known_points_px = [pt for pt in known_points_px if self._is_free_pixel(pt[0], pt[1])]

        self._debug_print(f"[DEBUG] Frame offsets: x_off={x_offset:.2f}, y_off={y_offset:.2f}")
        self._debug_print(f"[DEBUG] Known pts (world→pixel): {len(known_points_px)} visible this frame")
        self._debug_print(f"[DEBUG] New pts: {len(new_points_px)} total")

        if True:
            free_img = (inflated_for_skeleton * 255).astype(np.uint8)
            free_img = cv2.cvtColor(free_img, cv2.COLOR_GRAY2BGR)

            # color-code:
            #   known points → magenta
            #   free-space samples → red
            #   skeleton pixels (if any) → green overlay
            if hasattr(self, "_last_skeleton") and self._last_skeleton is not None:
                skel_vis = (self._last_skeleton > 0).astype(np.uint8)
                free_img[skel_vis == 1] = (0, 255, 0)

            # optionally overlay known points in magenta (same scale)
            if known_points_px is not None and len(known_points_px) > 0:
                for (y, x) in known_points_px:
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 0, 255), -1)

            # overlay persisted known nodes (magenta) ---
            for node, data in graph.nodes(data=True):
                ntype = str(data.get("node_type", "")).lower()
                if "known" in ntype:
                    pix = data.get("pixel")
                    if pix is None and isinstance(node, tuple) and len(node) == 2:
                        pix = node
                    if pix is None:
                        continue
                    y, x = pix
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 0, 255), -1)

            # overlay boundary nodes (cyan) ---
            for node, data in graph.nodes(data=True):
                ntype = str(data.get("node_type", "")).lower()
                if "boundary" in ntype:
                    y, x = node
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 255, 0), -1)

            self._debug_save_stage("05a_free_space_samples", free_img, color=True)

        # visualize new free-space points
        self._debug_print(f"[DEBUG free-space] total new nodes in graph after sampling: {len(new_points_px)}")

        if new_points_px is not None and len(new_points_px) > 0:
            free_img = (inflated_for_skeleton * 255).astype(np.uint8)
            free_img = cv2.cvtColor(free_img, cv2.COLOR_GRAY2BGR)

            # color-code:
            #   known points → magenta
            #   free-space samples → red
            #   skeleton pixels (if any) → green overlay
            if hasattr(self, "_last_skeleton") and self._last_skeleton is not None:
                skel_vis = (self._last_skeleton > 0).astype(np.uint8)
                free_img[skel_vis == 1] = (0, 255, 0)

            # draw free-space sample points as red dots
            for (y, x) in new_points_px:
                if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                    cv2.circle(free_img, (int(x), int(y)), 2, (0, 0, 255), -1)

            # optionally overlay known points in magenta (same scale)
            if known_points_px is not None and len(known_points_px) > 0:
                for (y, x) in known_points_px:
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 0, 255), -1)

            # overlay persisted known nodes (magenta) ---
            for node, data in graph.nodes(data=True):
                ntype = str(data.get("node_type", "")).lower()
                if "known" in ntype:
                    pix = data.get("pixel")
                    if pix is None and isinstance(node, tuple) and len(node) == 2:
                        pix = node
                    if pix is None:
                        continue
                    y, x = pix
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 0, 255), -1)

            # overlay boundary nodes (cyan) ---
            for node, data in graph.nodes(data=True):
                ntype = str(data.get("node_type", "")).lower()
                if "boundary" in ntype:
                    y, x = node
                    if 0 <= y < free_img.shape[0] and 0 <= x < free_img.shape[1]:
                        cv2.circle(free_img, (int(x), int(y)), 2, (255, 255, 0), -1)

            self._debug_save_stage("05_free_space_samples", free_img, color=True)
            self._debug_print(f"[DEBUG] Saved {len(new_points_px)} free-space samples to debug_stages/05_free_space_samples.png")
        else:
            self._debug_print("[DEBUG] No free-space samples to visualize.")

        boundary_nodes = sum(1 for _, d in graph.nodes(data=True) if d.get("node_type") == "boundary")
        free_nodes = sum(
            1
            for _, d in graph.nodes(data=True)
            if d.get("node_type") in {"free_space", "known"}
        )
        self._debug_print(f"[DEBUG] Node breakdown — boundary: {boundary_nodes}, free-space/known: {free_nodes}")

        # Remove skeleton nodes before running Delaunay
        skeleton_nodes = [
            n for n, data in graph.nodes(data=True)
            if str(data.get("node_type", "")).lower() == "skeleton"
        ]
        if skeleton_nodes:
            graph.remove_nodes_from(skeleton_nodes)
            self._logger.info(f"Removed {len(skeleton_nodes)} skeleton nodes before Delaunay.")

        # Lightly stitch inherited nodes to newly sampled ones before shortcuts
        self._connect_free_nodes(graph)

        # Drop any samples that landed on inflated obstacles/edges
        self._remove_invalid_free_nodes(graph)

        # Add shortcuts
        if self._config.use_delaunay_shortcuts:

            #edges_px = self.project_edges_to_current_frame(global_graph, inflated_for_skeleton.shape, x_offset, y_offset, rotation)
            #for (p1, p2) in edges_px:
            #    cv2.line(inflated_for_skeleton, (p1[1], p1[0]), (p2[1], p2[0]), color=255, thickness=1)

            for node in list(graph.nodes()):
                data = graph.nodes[node]
                if "pixel" not in data:
                    data["pixel"] = node  # assume (row, col)
                if "world" not in data:
                    y, x = node
                    x_w, y_w = self._pixel_to_world(y, x)
                    data["world"] = (x_w, y_w)

            self._add_delaunay_shortcuts(graph)
            final_debug = cv2.cvtColor((self._inflated_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for (n1, n2, data) in graph.edges(data=True):
                if "pixel" not in graph.nodes[n1] or "pixel" not in graph.nodes[n2]:
                    continue
                y1, x1 = map(int, graph.nodes[n1]["pixel"])
                y2, x2 = map(int, graph.nodes[n2]["pixel"])

                node1_type = str(graph.nodes[n1].get("node_type", "")).lower()
                node2_type = str(graph.nodes[n2].get("node_type", "")).lower()
                origin1 = graph.nodes[n1].get("origin")
                origin2 = graph.nodes[n2].get("origin")

                if "boundary" in (node1_type, node2_type):
                    color = (255, 255, 0)  # cyan for boundary edges
                elif origin1 == origin2 == "known":
                    color = (255, 0, 255)  # magenta for known↔known
                else:
                    color = (0, 0, 255)  # red for edges touching new nodes

                cv2.line(final_debug, (x1, y1), (x2, y2), color, 1)

            self._debug_save_stage("06_final_graph", final_debug, color=True)
            self._debug_print(f"[DEBUG] Delaunay shortcut edges added: {len(graph.edges)} total")

        # Add world coordinates to nodes
        #graph = self._to_world_coordinates(graph)
        self._debug_transform_vectors()

        # Modified --- ensures every node has "world"
        for _, data in graph.nodes(data=True):
            if "pixel" not in data and "world" in data:
                x_w, y_w = data["world"]
                y_p, x_p = self._world_to_pixel(Point(x=x_w, y=y_w, z=0))
                data["pixel"] = (y_p, x_p)
            elif "world" not in data and "pixel" in data:
                y_p, x_p = data["pixel"]
                x_w, y_w = self._pixel_to_world(y_p, x_p)
                data["world"] = (x_w, y_w)

            if "pos" not in data and "pixel" in data:
                data["pos"] = data["pixel"]
            if "node_type" not in data:
                data["node_type"] = "free_space"

        if len(graph.nodes) == 0:
            self._logger.warning("[WARN] Graph has no nodes after world/pixel consistency check.")
        else:
            self._logger.info(f"[INFO] Graph consistency OK: {len(graph.nodes)} nodes.")

        # Modified --- update known points (added here for iterative use)
        reusable_types = {"free_space", "known"}
        raw_points = [
            (float(n["world"][0]), float(n["world"][1]))
            for _, n in graph.nodes(data=True)
            if "world" in n and str(n.get("node_type", "")).lower() in reusable_types
        ]

        def _downsample(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
            grid: dict[tuple[int, int], tuple[float, float]] = {}
            for x, y in points:
                key = (int(math.floor(x / spacing)), int(math.floor(y / spacing)))
                if key not in grid:
                    grid[key] = (x, y)
            if len(grid) < len(points) * 0.9 and spacing < self._config.free_space_sampling_threshold:
                return _downsample(list(grid.values()), spacing * 1.05)
            return list(grid.values())

        spacing = self._config.free_space_sampling_threshold * 0.8
        self._known_points = _downsample(raw_points, spacing)

        # Prune unnecessary nodes and edges from the graph
        if self._config.prune_graph:
            self._prune_graph(graph)

        # Store and convert the graph to CSR format
        self._graph = graph
        self._logger.info(f"Final graph has {len(graph.nodes)} nodes and {len(graph.edges)} unique edges")

        # --- Clean up malformed pixel entries before building nearest node map ---
        fixed_count = 0
        removed_count = 0

        for node, data in self._graph.nodes(data=True):
            pix = data.get("pixel")
            if pix is None:
                if isinstance(node, tuple) and len(node) == 2:
                    data["pixel"] = node
                    fixed_count += 1
                continue

            pix_arr = np.array(pix).flatten()

            # Validate shape
            if pix_arr.shape[0] < 2:
                self._debug_print(f"[WARN] Dropping malformed pixel for node {node}: {pix}")
                data.pop("pixel", None)
                removed_count += 1
                continue

            # Assign clean 2D int tuple
            y, x = map(int, pix_arr[:2])
            data["pixel"] = (y, x)
            fixed_count += 1

        self._debug_print(f"[DEBUG] Normalized {fixed_count} pixels, removed {removed_count} malformed entries.")
        # -------------------------------------------------------------------------

        self._build_nearest_node_map(self._graph)
        return self._graph

    def project_edges_to_current_frame(self, graph: nx.Graph, inflated_map_shape, x_offset, y_offset, rotation=0.0):
        """
        Convert all world-space edges to current local pixel frame for visualization or masking.
        """
        edges_px = []

        for n1, n2, data in graph.edges(data=True):
            w1 = graph.nodes[n1].get("world")
            w2 = graph.nodes[n2].get("world")
            if w1 is None or w2 is None:
                continue

            # Build temporary Point objects from stored world coords
            p1_world = Point(x=w1[0], y=w1[1], z=0.0)
            p2_world = Point(x=w2[0], y=w2[1], z=0.0)

            # Temporarily override offsets/rotation for the current local frame
            prev_off_x, prev_off_y = self._x_offset, self._y_offset
            prev_cos, prev_sin = self._cos_rot, self._sin_rot

            self._x_offset = x_offset
            self._y_offset = y_offset
            self._cos_rot = np.cos(rotation)
            self._sin_rot = np.sin(rotation)

            try:
                pix1 = self._world_to_pixel(p1_world)
                pix2 = self._world_to_pixel(p2_world)
            finally:
                # restore transform parameters
                self._x_offset = prev_off_x
                self._y_offset = prev_off_y
                self._cos_rot = prev_cos
                self._sin_rot = prev_sin

            if pix1 is None or pix2 is None:
                continue

            row1, col1 = pix1
            row2, col2 = pix2

            # check bounds
            h, w = inflated_map_shape
            if not (0 <= row1 < h and 0 <= col1 < w and 0 <= row2 < h and 0 <= col2 < w):
                continue

            edges_px.append(((row1, col1), (row2, col2)))

        return edges_px

    def _debug_save_stage(self, name: str, image: np.ndarray, color: bool = False, graph=None):
        """Save intermediate debug images with consistent naming."""
        # Only save if debug flag is enabled
        if not self._config.debug:
            return
            
        import os
        os.makedirs("debug_stages", exist_ok=True)
        path = f"debug_stages/{name}.png"# Add shortcuts

        if image is None:
            self._logger.warning(f"[WARN] No image provided for {name}")
            return

        # Convert to color if needed for node overlays
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img = img.copy()

        # Optional: overlay graph nodes if provided
        if graph is not None:
            for node in graph.nodes():
                try:
                    y, x = map(int, node)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                except Exception:
                    continue

        cv2.imwrite(path, img)
        self._debug_print(f"[DEBUG] Saved {path}")

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
            # Modified: handles nodes as either tuple (row, col) or just int
            if isinstance(node, tuple):
                row, col = node
            else:
                row, col = data["pixel"]

            x_w, y_w = self._pixel_to_world(row, col)
            world_point_array = (x_w, y_w, 0.0)
            #world_point_array = tuple((world_point.x, world_point.y, world_point.z))
            pixel = (int(row), int(col))

            # Modified: fix for duplicated 'pixel'
            data_copy = data.copy()
            data_copy.pop("pixel", None)
            data_copy.pop("world", None)
            data_copy["pixel"] = pixel
            data_copy["world"] = world_point_array
            world_graph.add_node(i, **data_copy)
            pixel_to_id_lookup[node] = i

        # Convert edge weights from pixels to meters and copy edges
        for src, dst, data in graph.edges(data=True):
            edge_data = data.copy()
            if "weight" in edge_data:
                edge_data["weight"] = float(edge_data["weight"]) * self._resolution
            world_graph.add_edge(pixel_to_id_lookup[src], pixel_to_id_lookup[dst], **edge_data)

        return world_graph

    def _build_graph_from_skeleton(self, skeleton_sample_distance: int) -> np.ndarray:
        """Generate a graph from the skeleton of the inflated map."""
        self._logger.info("Generating skeleton...")
        try:
            skeleton_image = thin((1 - inflated_map).view(cp.uint8))
        except RuntimeError as e:
            from skimage.morphology import skeletonize

            self._logger.warning(f"Failed to generate skeleton with cucim: {e}. Falling back to skimage.")
            skeleton_image = skeletonize(1 - inflated_map)

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

        for node in graph.nodes():
            data = graph.nodes[node]
            data["node_type"] = "skeleton"
            if isinstance(node, tuple) and len(node) == 2:
                data.setdefault("pixel", (int(node[0]), int(node[1])))
            if "pixel" in data:
                row, col = data["pixel"]
                data["world"] = self._pixel_to_world(row, col)

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

        # --- NEW: handle world-coordinate inputs ---
        def ensure_pixel(pt):
            # If already pixel-like (ints), keep them
            if all(isinstance(v, (int, np.integer)) for v in pt):
                return pt
            # Otherwise convert from world → pixel
            if isinstance(pt, np.ndarray):
                x, y = pt
            else:
                x, y = pt[0], pt[1]
            row, col = self._world_to_pixel(Point(x=float(x), y=float(y), z=0.0))
            return (int(row), int(col))

        p0_px = ensure_pixel(p0)
        p1_px = ensure_pixel(p1)
        y0, x0 = p0_px
        y1, x1 = p1_px
        # --- rest of your collision logic below ---

        # Bresenham's line algorithm to get the pixels the line passes through
        #y0, x0 = p0
        #y1, x1 = p1
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

    def _free_mask_from_map(self, occupancy: np.ndarray) -> np.ndarray:
        """Return binary mask of free space (1 = free)."""
        if occupancy is None:
            raise RuntimeError("Occupancy map is not initialized.")
        if occupancy.max() <= 1:
            return (occupancy == 0).astype(np.uint8)
        return (occupancy > self._occupancy_threshold).astype(np.uint8)

    def _is_free_pixel(self, row: int, col: int) -> bool:
        if self._inflated_map is None:
            return True
        h, w = self._inflated_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self._inflated_map[row, col] == 0

    def _sample_free_space(self, graph, distance_threshold: float):
        """
        Iteratively sample free space in a map by identifying and adding nodes at local maxima
        of large distance areas until no such areas remain.

        Args:
            graph: The graph to which new nodes will be added.
            distance_threshold: The threshold to identify areas with large distances.
        """
        new_points = []
        base_map = self._inflated_map if self._inflated_map is not None else self._original_map
        free_mask = self._free_mask_from_map(base_map)

        distance_map = np.full(free_mask.shape, np.inf, dtype=np.float64)
        distance_map[free_mask == 0] = 0

        def _node_pixel(node, data):
            pixel = data.get("pixel")
            if pixel is None and isinstance(node, tuple) and len(node) == 2:
                pixel = node
            if pixel is None:
                return None
            row, col = int(pixel[0]), int(pixel[1])
            if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
                return row, col
            return None

        for node, data in graph.nodes(data=True):
            if str(data.get("node_type", "")).lower() == "skeleton":
                continue
            pix = _node_pixel(node, data)
            if pix is not None:
                distance_map[pix[0], pix[1]] = 0

        # Initialize kernel for dilation
        kernel = np.ones((3, 3), np.uint8)

        while True:
            # Use distance transform to calculate distances to the nearest node
            distance_map = cv2.distanceTransform(
                (distance_map > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )

            # --- DEBUG INSERTION ---
            if self._config.debug:
                self.last_distance_map = distance_map.copy()
                cv2.imwrite(
                    "debug_distance_map_free_space.png",
                    cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX),
                )
                self._debug_print("[DEBUG free_space] distance_map max (px):", float(distance_map.max()))
            # -----------------------

            # Identify large distance areas
            large_distance_areas = (distance_map > distance_threshold).astype(np.uint8)

            if not np.any(large_distance_areas):
                break

            # Dilate the large distance areas
            dilated = cv2.dilate(distance_map, kernel)

            # Find local maxima
            local_maxima_mask = (distance_map == dilated) & (large_distance_areas > 0)
            local_maxima_coords = np.column_stack(np.where(local_maxima_mask))
            self._debug_print(f"[DEBUG sampler] found {len(local_maxima_coords)} local maxima")

            idx = index.Index()
            idx_counter = 0
            for node, data in graph.nodes(data=True):
                if str(data.get("node_type", "")).lower() == "skeleton":
                    continue
                pix = _node_pixel(node, data)
                if pix is None:
                    continue
                row, col = pix
                idx.insert(idx_counter, (col, row, col, row))
                idx_counter += 1

            added = False
            for row, col in local_maxima_coords:
                if free_mask[row, col] == 0:
                    continue
                half_threshold = distance_threshold / 2
                bounding_box = (col - half_threshold, row - half_threshold, col + half_threshold, row + half_threshold)
                if list(idx.intersection(bounding_box)):
                    continue
                graph.add_node((row, col), node_type="free_space", pixel=(row, col), world=self._pixel_to_world(row, col))
                idx.insert(idx_counter, (col, row, col, row))
                idx_counter += 1
                distance_map[row, col] = 0
                new_points.append((row, col))
                added = True

            self._debug_print(f"[DEBUG sampler] distance_map max: {distance_map.max():.2f}, threshold(px): {distance_threshold}")

            if not added:
                break

        return new_points

    # Moddified --- added a section to account for known points
    def _sample_free_space_incremental(self, graph, known_points, distance_threshold):
        """
        Incrementally sample free-space areas by treating known points as existing samples,
        not as obstacles. This mirrors _sample_free_space() exactly, except it seeds the
        distance map with known points.
        """
 
        # --- Safety mask of free space ---
        if not hasattr(self, "_safe_mask") or self._safe_mask is None:
            self._safe_mask = (self._original_map > self._occupancy_threshold).astype(np.uint8)
        safe_mask = self._safe_mask
        H, W = safe_mask.shape    
        free_mask = safe_mask.copy()

        distance_map = np.full((H, W), np.inf, dtype=np.float64)
        distance_map[free_mask == 0] = 0

        def _node_pixel(node, data):
            pixel = data.get("pixel")
            if pixel is None and isinstance(node, tuple) and len(node) == 2:
                pixel = node
            if pixel is None:
                return None
            row, col = int(pixel[0]), int(pixel[1])
            if 0 <= row < H and 0 <= col < W:
                return row, col
            return None

        known_pixels = []
        for (x, y) in known_points:
            pix = self._world_to_pixel(Point(x=x, y=y, z=0.0))
            if pix is None:
                continue
            row, col = pix
            if 0 <= row < H and 0 <= col < W and free_mask[row, col] == 1:
                known_pixels.append((row, col))
                distance_map[row, col] = 0

        for node, data in graph.nodes(data=True):
            if str(data.get("node_type", "")).lower() == "skeleton":
                continue
            pix = _node_pixel(node, data)
            if pix is not None:
                distance_map[pix[0], pix[1]] = 0

        kernel = np.ones((3, 3), np.uint8)
        new_points = []

        while True:
            mask = free_mask.copy()
            for row, col in known_pixels:
                mask[row, col] = 0
            for node, data in graph.nodes(data=True):
                if str(data.get("node_type", "")).lower() == "skeleton":
                    continue
                pix = _node_pixel(node, data)
                if pix is not None:
                    mask[pix[0], pix[1]] = 0

            distance_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            large_distance_areas = (distance_map > distance_threshold).astype(np.uint8)
            if not np.any(large_distance_areas):
                self._logger.info("[INFO] No new free-space regions above threshold.")
                break

            dilated = cv2.dilate(distance_map, kernel)
            local_maxima_mask = (distance_map == dilated) & (large_distance_areas > 0)
            local_maxima_coords = np.column_stack(np.where(local_maxima_mask))

            idx = index.Index()
            idx_counter = 0
            for node, data in graph.nodes(data=True):
                if str(data.get("node_type", "")).lower() == "skeleton":
                    continue
                pix = _node_pixel(node, data)
                if pix is None:
                    continue
                row, col = pix
                idx.insert(idx_counter, (col, row, col, row))
                idx_counter += 1

            added = False
            for row, col in local_maxima_coords:
                if free_mask[row, col] == 0:
                    continue

                half_threshold = distance_threshold / 2
                bounding_box = (col - half_threshold, row - half_threshold, col + half_threshold, row + half_threshold)
                if list(idx.intersection(bounding_box)):
                    continue

                graph.add_node((row, col), pixel=(row, col), node_type="free_space", world=self._pixel_to_world(row, col))
                idx.insert(idx_counter, (col, row, col, row))
                idx_counter += 1
                new_points.append((row, col))
                distance_map[row, col] = 0
                added = True

            if not added:
                break

        return new_points

    def _remove_skeleton_nodes(self, graph: nx.Graph) -> None:
        """Remove skeleton nodes so they do not affect downstream shortcuts."""
        nodes_to_remove = [
            n for n, data in graph.nodes(data=True)
            if str(data.get("node_type", "")).lower() == "skeleton"
        ]
        if nodes_to_remove:
            graph.remove_nodes_from(nodes_to_remove)
            self._logger.info(f"Removed {len(nodes_to_remove)} skeleton nodes before Delaunay.")

    def _connect_free_nodes(self, graph: nx.Graph) -> None:
        """Connect nearby free-space/boundary nodes to reduce seams."""
        candidates: list = []
        coords: list[tuple[float, float]] = []

        for node, data in graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).lower()
            if node_type not in {"free_space", "boundary", "known"}:
                continue

            world = data.get("world")
            if world is None:
                pix = data.get("pixel")
                if pix is None:
                    continue
                world = self._point_to_tuple(self._pixel_to_world(pix[0], pix[1]))
                data["world"] = world

            world_tuple = self._point_to_tuple(world)
            candidates.append(node)
            coords.append((float(world_tuple[0]), float(world_tuple[1])))

        if len(coords) < 2:
            return

        try:
            tree = cKDTree(coords)
        except Exception:
            return

        radius = max(self._config.free_space_sampling_threshold * 1.5, self._config.merge_node_distance * 2.0)
        pairs = tree.query_pairs(r=radius)

        for i, j in pairs:
            n1 = candidates[i]
            n2 = candidates[j]
            if n1 == n2:
                continue

            w1 = np.asarray(self._point_to_tuple(graph.nodes[n1]["world"])[:2], dtype=float)
            w2 = np.asarray(self._point_to_tuple(graph.nodes[n2]["world"])[:2], dtype=float)
            dist = float(np.linalg.norm(w2 - w1))
            if not np.isfinite(dist) or dist <= 0.0:
                continue

            existing = graph.get_edge_data(n1, n2)
            if existing:
                current = existing.get("weight")
                if current is not None and current <= dist:
                    continue
                existing["weight"] = dist
                existing["edge_type"] = existing.get("edge_type", "known")
            else:
                graph.add_edge(n1, n2, weight=dist, edge_type="known")

    def _remove_invalid_free_nodes(self, graph: nx.Graph) -> None:
        if self._inflated_map is None:
            return

        height, width = self._inflated_map.shape
        to_remove: list = []
        for node, data in graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).lower()
            if node_type not in {"free_space", "known"}:
                continue

            pix = data.get("pixel")
            if pix is None:
                continue
            row, col = int(pix[0]), int(pix[1])
            if not (0 <= row < height and 0 <= col < width):
                to_remove.append(node)
                continue
            if self._inflated_map[row, col] != 0:
                to_remove.append(node)

        if to_remove:
            graph.remove_nodes_from(to_remove)

    def _add_delaunay_shortcuts(self, graph: nx.Graph):
        """
        Create shortcuts based on Delaunay triangulation of nodes.

        Args:
            graph: NetworkX graph to add shortcuts to
        """

        self._logger.info("Adding Delaunay shortcuts (world-frame)...")

        node_worlds = []
        idx_to_node = []

        # Collect world positions for all nodes
        for n, data in graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                pix = data.get("pixel")
                if pix is not None:
                    y, x = pix
                    point = self._pixel_to_world(y, x)
                    world = self._point_to_tuple(point)
                    data["world"] = world

            if world is None:
                continue

            world = self._point_to_tuple(world)
            if np.isfinite(world[0]) and np.isfinite(world[1]):
                node_worlds.append((world[0], world[1]))
                idx_to_node.append(n)

        if len(node_worlds) < 3:
            self._logger.warning("[WARN] Not enough nodes for Delaunay triangulation — skipping.")
            return

        node_coords = np.array(node_worlds, dtype=float)

        try:
            tri = Delaunay(node_coords)
        except Exception as e:
            self._logger.error(f"Skipping Delaunay triangulation: {e}")
            return

        edge_candidates = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = idx_to_node[simplex[i]]
                b = idx_to_node[simplex[(i + 1) % 3]]
                edge_candidates.add(tuple(sorted((a, b))))

        # Add edges if free of collision
        added = 0
        max_delaunay_distance = max(
            self._config.free_space_sampling_threshold * 1.5,  # Reduced from 2.0
            self._config.merge_node_distance * 2.5,  # Reduced from 3.0
        )

        for n1, n2 in edge_candidates:
            try:
                w1 = np.asarray(self._point_to_tuple(graph.nodes[n1]["world"])[:2], dtype=float)
                w2 = np.asarray(self._point_to_tuple(graph.nodes[n2]["world"])[:2], dtype=float)
                dist = float(np.linalg.norm(w2 - w1))
                if not np.isfinite(dist) or dist <= 0.0 or dist > max_delaunay_distance:
                    continue

                # For collision check, project back into pixel space of this map
                p1 = graph.nodes[n1].get("pixel")
                p2 = graph.nodes[n2].get("pixel")
                if p1 is None:
                    p1 = self._world_to_pixel(Point(x=w1[0], y=w1[1], z=0))
                if p2 is None:
                    p2 = self._world_to_pixel(Point(x=w2[0], y=w2[1], z=0))
                if p1 is None or p2 is None:
                    continue

                if not self._check_line_collision(p1, p2):
                    existing = graph.get_edge_data(n1, n2)
                    if existing:
                        current = existing.get("weight")
                        if current is not None and current <= dist:
                            continue
                        existing["weight"] = dist
                        existing["edge_type"] = "delaunay"
                        added += 1
                    else:
                        graph.add_edge(n1, n2, weight=dist, edge_type="delaunay")
                        added += 1
            except Exception:
                continue

        self._logger.info(f"[INFO] Delaunay shortcuts added: {added}")


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
                graph.add_node(
                    (row_1, col_1),
                    node_type="boundary",
                    pixel=(row_1, col_1),
                    world=self._pixel_to_world(row_1, col_1),
                )

                # Calculate distance to next vertex
                segment_length = np.linalg.norm(p2 - p1)
                # Add intermediate points
                num_intermediate = int(segment_length / sample_distance)
                # Skip first and last points
                intermediate_points = np.linspace(p1, p2, num=num_intermediate, endpoint=False).astype(int).tolist()[1:]
                for point in intermediate_points:
                    # Interpolate point
                    col, row = point
                    contour_nodes.append((row, col))
                    graph.add_node(
                        (row, col),
                        node_type="boundary",
                        pixel=(row, col),
                        world=self._pixel_to_world(row, col),
                    )

            # Connect consecutive nodes along this contour
            if len(contour_nodes) > 1:
                self._connect_contour_nodes(contour_nodes, graph)

        num_nodes_added = len(graph.nodes()) - initial_num_nodes
        self._logger.info(f"Added {num_nodes_added} nodes along obstacle boundaries")

    def _find_obstacle_contours(self, boundary_inflation: float) -> list[np.ndarray]:
        """Find contours of inflated obstacles using the distance map."""

        filtered_obstacles = (self._dist_transform >= boundary_inflation).astype(np.uint8)
        contours, _ = cv2.findContours(filtered_obstacles, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        self._debug_print(f"[DEBUG] Found {len(contours)} contours from distance transform.")
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
            # Merge nodes that are close to each other (threshold expressed in meters)
            self._merge_close_nodes(graph, threshold=self._config.merge_node_distance)

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
        """Merge nodes that are within a certain distance threshold (meters)."""

        # --- SAFETY PATCH: ensure all nodes have 'pos' defined ---
        for node, data in graph.nodes(data=True):
            if "world" in data and isinstance(data["world"], (tuple, list)) and len(data["world"]) == 2:
                data["pos"] = np.array(data["world"], dtype=float)
            elif "pixel" in data and isinstance(data["pixel"], (tuple, list)) and len(data["pixel"]) == 2:
                # convert pixel → world explicitly
                y, x = data["pixel"]
                x_w, y_w = self._pixel_to_world(y, x)
                data["pos"] = np.array((x_w, y_w), dtype=float)
            else:
                # fallback to node key itself
                data["pos"] = np.array(node, dtype=float)
        # ----------------------------------------------------------

        while True:
            nodes = list(graph.nodes())

            # Modified - debug before node_coords
            self._debug_print("[DEBUG] Checking node positions before merging...")
            for i, n in enumerate(graph.nodes):
                pos = graph.nodes[n].get("pos", None)
                #print(f"{i:03d} {n}: {pos}")
            self._debug_print("[DEBUG] Total nodes:", len(graph.nodes))
            self._debug_print("[DEBUG] Threshold:", threshold)

            node_coords = [data["pos"] for _, data in graph.nodes(data=True) if "pos" in data]

            self._debug_print("[DEBUG] Building KDTree with node_coords:")
            self._debug_print(f"  Number of coords: {len(node_coords)}")

            bad_entries = []
            for i, coord in enumerate(node_coords):
                if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                    bad_entries.append((i, coord))
                    continue
            # ensure they are numeric
                try:
                    float(coord[0])
                    float(coord[1])
                except Exception as e:
                    bad_entries.append((i, coord))

            if bad_entries:
                self._debug_print("[ERROR] Found malformed coordinates:")
                for idx, bad in bad_entries:
                    #print(f"  Index {idx}: {bad} (type: {type(bad)})")
                    continue
            else:
                self._debug_print("[DEBUG] All coords look valid. Example few:", node_coords[:5])

            #node_coords = np.array(nodes)
            node_coords = np.array([[float(x), float(y)] for x, y in node_coords], dtype=float)
            self._debug_print(f"[DEBUG] node_coords.shape after conversion: {node_coords.shape}, dtype: {node_coords.dtype}")

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
                if not graph.has_node(n1) or not graph.has_node(n2):
                    continue

                # Coordinates of n1 and n2
                n1_pos = graph.nodes[n1]["pos"]
                n2_pos = graph.nodes[n2]["pos"]

                # Merge n2 into n1 and update edges
                # Check if all n2's neighbors can connect to n1 without collision
                neighbors = [n for n in graph.neighbors(n2) if n != n1 and graph.has_node(n)]
                if neighbors:
                    collision = False
                    for neighbor in neighbors:
                        neighbor_pos = graph.nodes[neighbor]["pos"]
                        if self._check_line_collision(n1_pos, neighbor_pos):
                            collision = True
                            break

                    if not collision:
                        for neighbor in neighbors:
                            neighbor_pos = graph.nodes[neighbor]["pos"]
                            dist = np.linalg.norm(np.array(neighbor_pos) - np.array(n1_pos))
                            graph.add_edge(n1, neighbor, weight=dist, edge_type="merge")
                        graph.remove_node(n2)
                        merged = True
                else:
                    # No neighbors to check, safe to remove
                    graph.remove_node(n2)
                    merged = True  # A merge occurred

            if not merged:
                break  # Exit loop if no merges occurred in this iteration

    def _pixel_to_world(self, row: float, col: float) -> Point:
        """Convert pixel coordinates to world coordinates with the current transform."""
        return pixel_to_world(row, col, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot, self._image_shape)

    def _world_to_pixel(self, point: Point) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates with the inverse transform."""
        return world_to_pixel(point, self._resolution, self._x_offset, self._y_offset, self._cos_rot, self._sin_rot, self._image_shape)

    def _point_to_tuple(self, value) -> tuple[float, float, float]:
        """Normalize Point-like inputs to an (x, y, z) tuple."""
        if hasattr(value, "x") and hasattr(value, "y"):
            x = float(value.x)
            y = float(value.y)
            z = float(getattr(value, "z", 0.0))
            return (x, y, z)
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
            x = float(value[0])
            y = float(value[1])
            z = float(value[2]) if len(value) >= 3 else 0.0
            return (x, y, z)
        # Fallback: treat as origin
        return (0.0, 0.0, 0.0)

    def _debug_transform_vectors(self):
        """Print sanity checks for world↔pixel transforms."""
        self._debug_print("=== DEBUG TRANSFORM VECTORS ===")

        test_points = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1)
        ]

        for (x, y) in test_points:
            px = self._world_to_pixel((x, y))
            if px is None:
                self._debug_print(f"World ({x:.2f}, {y:.2f}) -> OUT OF FRAME")
                continue

            # px = (row, col) = (y, x)
            try:
                wx = self._pixel_to_world(px[1], px[0])
                self._debug_print(f"World ({x:.2f}, {y:.2f}) -> Pixel {px} -> Reconstructed {wx}")
            except Exception as e:
                self._debug_print(f"[WARN] Failed to reconstruct for ({x:.2f}, {y:.2f}): {e}")

    def transform_known_points(known_points_world, dx, dy, theta, to_pixel_fn):
        """
        Apply frame-to-frame translation and rotation to known world points, then
        convert to pixel coordinates for plotting.
        Args:
            known_points_world: list of (x, y) in world coordinates (previous frame)
            dx, dy: translation (meters) between frames
            theta: rotation (radians, positive CCW)
            to_pixel_fn: function handle like self._world_to_pixel

        Returns:
            known_points_px: list of (x_px, y_px) in pixel coords for the current frame
        """
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        known_points_px = []

        for x, y in known_points_world:
            # Rotate about origin (0,0)
            x_rot = cos_t * x - sin_t * y
            y_rot = sin_t * x + cos_t * y

            # Translate
            x_new = x_rot + dx
            y_new = y_rot + dy

            # Convert to pixel space
            known_points_px.append(to_pixel_fn((x_new, y_new)))

        return known_points_px

    # Modified - visualize the graph, and nodes
    def visualize_graph(
        self,
        output_dir: str = ".",
        output_filename: str = "waypoint_graph.png",
    ) -> None:
        """
        Create and save a visualization of the graph with node IDs.
        Uses the existing map, node_map, and config colors.
        """
        if self._graph is None:
            raise RuntimeError("No graph has been built yet")

        self._logger.info("Starting visualization...")

        # Convert map to color
        map_vis = cv2.cvtColor(self._original_map, cv2.COLOR_GRAY2BGR)

        # Ensure nearest node map exists
        if self._node_map is None:
            self._build_nearest_node_map(self._graph)

        # --- Draw edges ---
        edge_color = self._config.edge_color.to_tuple()
        for u, v in self._graph.edges():
            try:
                src_y, src_x = self._graph.nodes[u]["pixel"]
                dst_y, dst_x = self._graph.nodes[v]["pixel"]

                cv2.line(map_vis, (src_x, src_y), (dst_x, dst_y), edge_color, 1)
            except KeyError:
                continue

        # --- Optional: overlay node map mask (for debug) ---
        overlay = cv2.applyColorMap(
            ((self._node_map > -1).astype(np.uint8) * 180).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        map_vis = cv2.addWeighted(map_vis, 0.8, overlay, 0.2, 0)

        # --- Save output ---
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        if not cv2.imwrite(output_path, map_vis):
            raise RuntimeError(f"Failed to save visualization to {output_path}")

        self._logger.info(f"Saved visualization with node IDs to {output_path}")


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

       # --- SAFETY NORMALIZATION ---
        cleaned_nodes = []
        for i, (node, pix) in enumerate(graph.nodes(data="pixel")):
            if pix is None:
                # fallback: use node key if it looks like coordinates
                if isinstance(node, tuple) and len(node) == 2:
                    y, x = node
                else:
                    self._debug_print(f"[WARN] Node {node} missing pixel info — skipped.")
                    continue
            elif isinstance(pix, np.ndarray):
                arr = np.array(pix).flatten()
                if arr.size < 2:
                    skipped += 1
                    continue
                y, x = arr[:2]
            elif isinstance(pix, list):
                # handle [[y, x]] or [y, x]
                if len(pix) == 1 and isinstance(pix[0], (list, tuple)):
                    y, x = pix[0][:2]
                else:
                    y, x = pix[:2]
            elif isinstance(pix, tuple) and len(pix) >= 2:
                y, x = pix[:2]
            else:
                self._debug_print(f"[WARN] Unexpected pixel format for node {node}: {pix}")
                continue

            try:
                y, x = pix[:2]
            except Exception as e:
                self._debug_print(f"[ERROR] Could not clean node {node}: {pix} ({e})")
                continue

            cleaned_nodes.append((i, y * width + x))
        # --------------------------------

        if not cleaned_nodes:
            self._logger.warning("[WARN] No valid pixel nodes for nearest-node map.")
            return

        graph_pixels = np.array(cleaned_nodes, dtype=np.int32)
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


    def _debug_plot_with_known_points(self, known_points_px=None, new_points_px=None, title="Debug Known Points", save_path="debug_plots"):
        """
        Save a debug visualization of the map, inflated map, and any known points overlaid.
        Works in SSH/headless mode (no GUI).

        Args:
            known_points_px: list of (x_px, y_px) pixel coordinates to plot (optional)
            title: figure title
            save_path: directory to save the plot
        """
        import matplotlib.pyplot as plt

        # Filter out any None or malformed entries
        known_points_px = [p for p in known_points_px if p is not None and len(p) == 2]
        new_points_px = [p for p in new_points_px if p is not None and len(p) == 2]

        if len(known_points_px) == 0 and len(new_points_px) == 0:
            self._logger.warning("[WARN] No valid points to plot in debug visualization.")
            return

        ys = [p[0] for p in known_points_px]
        xs = [p[1] for p in known_points_px]
        new_ys = [p[0] for p in new_points_px]
        new_xs = [p[1] for p in new_points_px]

        plt.figure(figsize=(6, 6))
        plt.imshow(self._original_map, cmap='gray')
        plt.scatter(xs, ys, c='lime', s=20, label='Known Points')
        plt.scatter(new_xs, new_ys, c='red', s=15, label='New Points')
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        save_path = f"debug_stages/{title}.png"
        plt.savefig(save_path)
        plt.close()
        self._debug_print(f"[DEBUG] Saved {save_path}")
