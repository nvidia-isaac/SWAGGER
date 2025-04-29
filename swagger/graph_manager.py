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

import os
import threading
import traceback

import networkx as nx
import numpy as np
from numpydantic import NDArray, Shape

from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig
from swagger.logger import Logger
from swagger.models import Point


class GraphManager:
    """Manages multiple WaypointGraphGenerator instances indexed by map ID."""

    def __init__(self, visualization_dir: str = "."):
        self._generators: dict[str, WaypointGraphGenerator] = {}
        self._lock = threading.Lock()  # Thread safety for concurrent access
        self._logger = Logger("graph_manager")
        self._visualization_dir = visualization_dir
        self._logger.info(f"Attempting to create visualization directory: {visualization_dir}")
        os.makedirs(visualization_dir, exist_ok=True)
        # Remove any existing PNG files in visualization directory
        self._logger.info(f"Removing existing waypoint graph visualization files in directory: {visualization_dir}")
        for file in os.listdir(visualization_dir):
            if file.lower().startswith("waypoint_graph_") and file.lower().endswith(".png"):
                os.remove(os.path.join(visualization_dir, file))

    def create_graph(
        self,
        map_id: str,
        image: NDArray[Shape["* y, * x"], np.uint8],  # noqa: F821
        resolution: float,
        safety_distance: float,
        occupancy_threshold: int = 127,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        rotation: float = 0.0,
        config: WaypointGraphGeneratorConfig = WaypointGraphGeneratorConfig(),
    ) -> nx.Graph:
        """
        Create a new graph for the given map ID or update existing one.

        Args:
            map_id: Unique identifier for the map
            image: Occupancy grid (0-255, where values <= threshold are occupied)
            resolution: Meters per pixel
            safety_distance: Robot radius in meters
            occupancy_threshold: Threshold to determine occupied cells (0-255)
            x_offset: Translation in x direction (meters)
            y_offset: Translation in y direction (meters)
            rotation: Rotation about Z axis (radians)

        Returns:
            NetworkX Graph object containing the waypoint graph
        """
        with self._lock:
            # Remove any existing visualization for this map_id
            self._remove_visualization(map_id)

            # Create new generator and graph
            generator = WaypointGraphGenerator(config=config)
            # Update or create new entry
            self._generators[map_id] = generator
            try:
                graph = generator.build_graph_from_grid_map(
                    image=image,
                    resolution=resolution,
                    safety_distance=safety_distance,
                    occupancy_threshold=occupancy_threshold,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    rotation=rotation,
                )
                return graph
            except ValueError as e:
                self._logger.error(f"Error creating graph for map_id: {map_id}, exception: {e}")
                self._logger.error(f"Traceback: {traceback.format_exc()}")
                raise e

    def get_graph(self, map_id: str) -> nx.Graph | None:
        """
        Get the graph for the given map ID.

        Args:
            map_id: Map identifier

        Returns:
            CsrGraph object if found, None otherwise
        """
        with self._lock:
            generator = self._generators.get(map_id)
            if generator is None:
                return None
            return generator._graph

    def get_nearest_nodes(self, map_id: str, points: list[Point]) -> list[int | None]:
        """
        Find nearest graph nodes to the given points.

        Args:
            map_id: Map identifier
            points: List of query points

        Returns:
            List of node indices (None for points with no valid path)

        Raises:
            KeyError: If map_id does not exist
        """
        with self._lock:
            generator = self._generators.get(map_id)
            if generator is None:
                raise KeyError(f"No graph found for map_id: {map_id}")
            return generator.get_node_ids(points)

    def find_route(self, map_id: str, start: Point, goal: Point) -> list[Point]:
        """
        Find the shortest route from a start to a goal in the real world.

        Args:
            map_id: Map identifier
            start: Start position in the real world
            goal: Goal position in the real world

        Returns:
            List of points representing the path (empty if no path found)

        Raises:
            KeyError: If map_id does not exist
        """
        with self._lock:
            generator = self._generators.get(map_id)
            if generator is None:
                raise KeyError(f"No graph found for map_id: {map_id}")
            return generator.find_route(start, goal)

    def get_visualization_path(self, map_id: str) -> str:
        """Get the path where the visualization image for this map should be stored."""
        return os.path.join(self._visualization_dir, f"waypoint_graph_{map_id}.png")

    def _remove_visualization(self, map_id: str) -> None:
        """Remove any existing visualization for the given map_id."""
        vis_path = self.get_visualization_path(map_id)
        if os.path.exists(vis_path):
            os.remove(vis_path)

    def visualize_graph(self, map_id: str) -> bool:
        """Create visualization of the graph for the given map ID."""
        with self._lock:
            generator = self._generators.get(map_id)
            if generator is None or generator._graph is None:
                return False

            # Check if visualization already exists
            vis_path = self.get_visualization_path(map_id)
            if os.path.exists(vis_path):
                return True

            # Create binary occupancy grid from original map using same threshold
            generator.visualize_graph(
                output_dir=self._visualization_dir,
                output_filename=f"waypoint_graph_{map_id}.png",
            )
            return True

    def remove_graph(self, map_id: str) -> bool:
        """
        Remove the graph and its visualization for the given map ID.

        Args:
            map_id: Map identifier

        Returns:
            True if graph was removed, False if map_id not found
        """
        with self._lock:
            if map_id in self._generators:
                self._remove_visualization(map_id)
                del self._generators[map_id]
                return True
            return False

    def clear(self) -> None:
        """Remove all graphs and their visualizations."""
        with self._lock:
            # Remove all visualizations
            for map_id in self._generators.keys():
                self._remove_visualization(map_id)
            self._generators.clear()

    def errors(self) -> dict[str, str]:
        """Return a dictionary of errors from all generators."""
        errors = {}
        with self._lock:
            for map_id, generator in self._generators.items():
                if generator.graph is None:
                    errors[f"graph_{map_id}"] = f"No graph has been built for map_id: {map_id}"
        return errors
