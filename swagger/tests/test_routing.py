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
import shutil
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig
from swagger.models import Point


class TestFindRoute(unittest.TestCase):
    """Test cases for the find_route method using real maps."""

    def setUp(self):
        """Set up the test configuration."""
        self.config = WaypointGraphGeneratorConfig(
            skeleton_sample_distance=1.0,
            boundary_sample_distance=1.0,
            free_space_sampling_threshold=2.0,
            merge_node_distance=0.3,
            min_subgraph_length=0.3,
        )

        # Resolution and robot parameters
        self.resolution = 0.1  # 10 cm per pixel
        self.safety_distance = 0.2  # 20 cm radius

        # Create a temporary directory for visualizations
        self.temp_dir = tempfile.mkdtemp(prefix="wpg_test_", dir="/tmp")

    def tearDown(self):
        """Clean up after test."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_path_around_box(self):
        """Test finding a path around a simple box obstacle."""
        # Create a simple map with a box obstacle in the middle
        map_size = 100
        grid_map = np.ones((map_size, map_size), dtype=np.uint8) * 255

        # Add a box obstacle in the middle
        grid_map[40:60, 40:60] = 0

        # Create the waypoint graph generator and build the graph
        wpg = WaypointGraphGenerator(config=self.config)
        wpg.build_graph_from_grid_map(
            image=grid_map, resolution=self.resolution, safety_distance=self.safety_distance, occupancy_threshold=127
        )

        # Save the visualization for debugging
        wpg.visualize_graph(self.temp_dir, "test_box_obstacle.png")

        # Define start and goal points on opposite sides of the obstacle
        start_point = Point(x=2.0, y=2.0, z=0.0)  # Top-left
        goal_point = Point(x=8.0, y=8.0, z=0.0)  # Bottom-right

        route = wpg.find_route(start_point, goal_point)

        self.assertGreater(len(route), 0, "No route found around box obstacle")

        # Verify the route goes around the obstacle by checking it's longer than direct path
        route_length = self._calculate_path_length(route)
        direct_distance = np.sqrt((goal_point.x - start_point.x) ** 2 + (goal_point.y - start_point.y) ** 2)

        self.assertGreater(
            route_length,
            direct_distance * 1.2,  # At least 20% longer
            "Route appears to go through the obstacle (too short)",
        )

        # Visualize the route for debugging
        self._visualize_path(grid_map, route, "test_box_path.png")

    def test_no_path_available(self):
        """Test the case where no path is available between start and goal."""
        # Create a map with a wall dividing it
        map_size = 100
        grid_map = np.ones((map_size, map_size), dtype=np.uint8) * 255

        # Add a wall across the entire map
        grid_map[0:100, 50:52] = 0

        # Create the waypoint graph generator and build the graph
        wpg = WaypointGraphGenerator(config=self.config)
        wpg.build_graph_from_grid_map(
            image=grid_map, resolution=self.resolution, safety_distance=self.safety_distance, occupancy_threshold=127
        )

        # Save the visualization for debugging
        wpg.visualize_graph(self.temp_dir, "test_no_path.png")

        # Define start and goal points on opposite sides of the wall
        start_point = Point(x=2.0, y=5.0, z=0.0)  # Left side
        goal_point = Point(x=8.0, y=5.0, z=0.0)  # Right side

        # Find a route
        route = wpg.find_route(start_point, goal_point)

        # Check that no route was found
        self.assertEqual(len(route), 0, "Route found when none should be possible")

    def _calculate_path_length(self, route: list[Point]) -> float:
        """Calculate the total length of a path."""
        total_length = 0.0
        for i in range(1, len(route)):
            segment_length = np.sqrt((route[i].x - route[i - 1].x) ** 2 + (route[i].y - route[i - 1].y) ** 2)
            total_length += segment_length
        return total_length

    def _visualize_path(self, grid_map, route: list[Point], filename: str):
        """Visualize a path on the map for debugging."""
        # Convert route points to pixel coordinates
        pixel_route = []
        for point in route:
            # Simple conversion assuming origin at (0,0) and no rotation
            x_pixel = int(point.x / self.resolution)
            y_pixel = int(point.y / self.resolution)
            pixel_route.append((x_pixel, y_pixel))

        # Create an RGB visualization
        vis_map = np.stack([grid_map, grid_map, grid_map], axis=2)

        # Create a new figure
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_map)

        # Draw the path in red
        if len(pixel_route) > 1:
            x_coords = [x for x, y in pixel_route]
            y_coords = [y for x, y in pixel_route]
            plt.plot(x_coords, y_coords, "r-", linewidth=2)
            plt.scatter(x_coords, y_coords, c="blue", s=30)

        # Mark start and end points
        if pixel_route:
            start_x, start_y = pixel_route[0]
            end_x, end_y = pixel_route[-1]
            plt.scatter([start_x], [start_y], c="green", s=100, marker="o")
            plt.scatter([end_x], [end_y], c="purple", s=100, marker="x")

        # Save the visualization
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, filename))
        plt.close()


if __name__ == "__main__":
    unittest.main()
