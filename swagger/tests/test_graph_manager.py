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
import tempfile
import unittest

import networkx as nx
import numpy as np

from swagger.graph_manager import GraphManager
from swagger.models import Point


class TestGraphManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for visualizations
        self.test_dir = tempfile.mkdtemp()
        self.manager = GraphManager(visualization_dir=self.test_dir)

        # Create a simple test map (20x20 with a wall in the middle)
        self.test_map = np.ones((20, 20), dtype=np.uint8) * 255
        self.test_map[8:12, :] = 0  # Add horizontal wall

        self.test_resolution = 0.05  # 5cm per pixel
        self.test_safety_distance = 0.2  # 20cm radius

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_create_graph(self):
        """Test creating a new graph."""
        map_id = "test_map"
        graph = self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Verify graph was created
        self.assertIsNotNone(graph)
        self.assertTrue(len(graph.nodes) > 0)
        self.assertTrue(len(graph.edges) > 0)

        # Verify graph is stored in manager
        self.assertIn(map_id, self.manager._generators)

    def test_get_graph(self):
        """Test retrieving an existing graph."""
        map_id = "test_map"

        # Create graph
        original_graph = self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Retrieve graph
        retrieved_graph = self.manager.get_graph(map_id)

        # Verify graphs match
        self.assertTrue(nx.is_isomorphic(original_graph, retrieved_graph))

    def test_get_nonexistent_graph(self):
        """Test retrieving a non-existent graph returns None."""
        self.assertIsNone(self.manager.get_graph("nonexistent_map"))

    def test_get_nearest_nodes(self):
        """Test finding nearest nodes to query points."""
        map_id = "test_map"

        # Create graph
        graph = self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        for i, (_, world_coords) in enumerate(graph.nodes(data="world")):
            node_id = self.manager.get_nearest_nodes(map_id, [Point(x=world_coords[0], y=world_coords[1], z=0.0)])[0]
            self.assertEqual(node_id, i, f"Node {i} should have id {i} but got {node_id}")

        # Calculate map dimensions in world coordinates
        map_height = self.test_map.shape[0] * self.test_resolution
        map_width = self.test_map.shape[1] * self.test_resolution

        # Test points
        test_points = [
            Point(x=0.25, y=0.25, z=0.0),  # Inside map, away from wall
            Point(x=0.25, y=0.25, z=0.0),  # Duplicate point
            Point(x=map_width + 1.0, y=map_height + 1.0, z=0.0),  # Outside map
        ]

        # Get nearest nodes
        node_indices = self.manager.get_nearest_nodes(map_id, test_points)

        # Verify results
        self.assertEqual(len(node_indices), len(test_points))
        self.assertIsNotNone(node_indices[0])  # Should find a node for first point
        self.assertIsNotNone(node_indices[1])  # Should find a node for second point
        self.assertEqual(node_indices[0], node_indices[1])  # Should find same node for duplicate points
        self.assertIsNone(node_indices[2])  # Should not find node for point outside map

    def test_get_nearest_nodes_nonexistent_map(self):
        """Test finding nearest nodes for non-existent map raises KeyError."""
        with self.assertRaises(KeyError):
            self.manager.get_nearest_nodes("nonexistent_map", [Point(x=0.0, y=0.0, z=0.0)])

    def test_visualize_graph(self):
        """Test graph visualization."""
        map_id = "test_map"

        # Create graph
        self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Generate visualization
        success = self.manager.visualize_graph(map_id)
        self.assertTrue(success)

        # Verify visualization file exists
        vis_path = self.manager.get_visualization_path(map_id)
        self.assertTrue(os.path.exists(vis_path))

        # Verify visualization is not regenerated if it exists
        mtime = os.path.getmtime(vis_path)
        success = self.manager.visualize_graph(map_id)
        self.assertTrue(success)
        self.assertEqual(mtime, os.path.getmtime(vis_path))

    def test_visualize_nonexistent_graph(self):
        """Test visualizing non-existent graph returns False."""
        self.assertFalse(self.manager.visualize_graph("nonexistent_map"))

    def test_remove_graph(self):
        """Test removing a graph."""
        map_id = "test_map"

        # Create graph and visualization
        self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )
        self.manager.visualize_graph(map_id)

        # Verify graph and visualization exist
        vis_path = self.manager.get_visualization_path(map_id)
        self.assertTrue(os.path.exists(vis_path))
        self.assertIn(map_id, self.manager._generators)

        # Remove graph
        success = self.manager.remove_graph(map_id)
        self.assertTrue(success)

        # Verify graph and visualization are removed
        self.assertFalse(os.path.exists(vis_path))
        self.assertNotIn(map_id, self.manager._generators)

    def test_remove_nonexistent_graph(self):
        """Test removing non-existent graph returns False."""
        self.assertFalse(self.manager.remove_graph("nonexistent_map"))

    def test_clear(self):
        """Test clearing all graphs."""
        # Create multiple graphs and visualizations
        map_ids = ["map1", "map2", "map3"]
        for map_id in map_ids:
            self.manager.create_graph(
                map_id=map_id,
                image=self.test_map,
                resolution=self.test_resolution,
                safety_distance=self.test_safety_distance,
            )
            self.manager.visualize_graph(map_id)

        # Verify graphs and visualizations exist
        for map_id in map_ids:
            vis_path = self.manager.get_visualization_path(map_id)
            self.assertTrue(os.path.exists(vis_path))
            self.assertIn(map_id, self.manager._generators)

        # Clear all graphs
        self.manager.clear()

        # Verify all graphs and visualizations are removed
        for map_id in map_ids:
            vis_path = self.manager.get_visualization_path(map_id)
            self.assertFalse(os.path.exists(vis_path))
            self.assertNotIn(map_id, self.manager._generators)
        self.assertEqual(len(self.manager._generators), 0)

    def test_errors(self):
        """Test error reporting."""
        # Initially should have no errors
        self.assertEqual(len(self.manager.errors()), 0)

        # Create a graph
        map_id = "test_map"
        self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Should still have no errors after successful graph creation
        self.assertEqual(len(self.manager.errors()), 0, self.manager.errors())

        # Create a graph with potential issues (empty map)
        empty_map_id = "empty_map"
        empty_map = np.ones((1, 1), dtype=np.uint8) * 255
        with self.assertRaises(ValueError):
            self.manager.create_graph(
                map_id=empty_map_id,
                image=empty_map,
                resolution=self.test_resolution,
                safety_distance=self.test_safety_distance,
            )

        # Check errors
        errors = self.manager.errors()
        self.assertGreaterEqual(len(errors), 1)  # May have errors for empty map

    def test_create_graph_with_threshold(self):
        """Test creating a graph with different occupancy thresholds."""
        # Create test map with three intensity regions
        test_map = np.ones((30, 30), dtype=np.uint8) * 127  # Middle intensity
        test_map[5:25, 5:25] = 200  # Large free area with high threshold
        test_map[10:20, 10:20] = 50  # Obstacle area with low threshold

        # Test with high threshold (more obstacles)
        high_graph = self.manager.create_graph(
            map_id="test_map_high",
            image=test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
            occupancy_threshold=210,  # High threshold means more cells are obstacles
        )

        # Test with low threshold (fewer obstacles)
        low_graph = self.manager.create_graph(
            map_id="test_map_low",
            image=test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
            occupancy_threshold=30,  # Low threshold means fewer cells are obstacles
        )

        # Low threshold should result in more nodes (fewer obstacles)
        self.assertGreater(len(low_graph.nodes), len(high_graph.nodes))

    def test_create_graph_with_transform(self):
        """Test creating a graph with transform parameters."""
        # Create test map
        test_map = np.zeros((20, 20), dtype=np.uint8)
        test_map[8:12, :] = 255  # Horizontal corridor

        # Set up transform
        x_offset = 1.0
        y_offset = 2.0
        rotation = np.pi / 4  # 45 degrees

        # Create graph with transform
        graph = self.manager.create_graph(
            map_id="test_map",
            image=test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )

        # Verify nodes are transformed
        for node in graph.nodes:
            # All nodes should be offset by at least the translation
            self.assertGreaterEqual(node.x, x_offset - 0.1)  # Allow small numerical error
            self.assertGreaterEqual(node.y, y_offset - 0.1)

    def test_nearest_nodes_with_transform(self):
        """Test finding nearest nodes with transformed graph."""
        # Create graph with transform
        x_offset = 1.0
        y_offset = 2.0
        rotation = np.pi / 4  # 45 degrees

        map_id = "test_map"
        graph = self.manager.create_graph(
            map_id=map_id,
            image=self.test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )

        # Get a valid node from the graph to ensure we query a point near it
        first_node = graph.nodes[0]["world"]

        # Query points in transformed space, near an existing node
        test_points = [
            Point(
                x=first_node[0] - 0.5 * self.test_resolution, y=first_node[1] - 0.5 * self.test_resolution, z=0.0
            ),  # Small offset from known node
            Point(x=x_offset + 10.0, y=y_offset + 10.0, z=0.0),  # Point far outside map
        ]

        # Get nearest nodes
        node_indices = self.manager.get_nearest_nodes(map_id, test_points)

        # Verify results
        self.assertEqual(len(node_indices), len(test_points))
        self.assertIsNotNone(node_indices[0])  # Should find a node for first point
        self.assertIsNone(node_indices[1])  # Should not find node for point outside map

    def test_find_route(self):
        """Test finding a route between two points."""
        map_id = "test_map"

        # Create a simple map with two clear areas separated by a wall with a gap
        route_test_map = np.ones((20, 20), dtype=np.uint8) * 255
        route_test_map[8:12, :] = 0  # Add horizontal wall
        route_test_map[8:12, 5:15] = 255  # Add a gap in the wall

        # Create graph
        self.manager.create_graph(
            map_id=map_id,
            image=route_test_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Define start and goal points on opposite sides of the wall
        start = Point(x=0.25, y=0.25, z=0.0)  # Top side
        goal = Point(x=0.25, y=0.75, z=0.0)  # Bottom side

        # Find route
        route = self.manager.find_route(map_id, start, goal)

        # Verify route exists and has reasonable properties
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 0)

        # Verify start and end points are close to requested points
        start_distance = np.sqrt((route[0].x - start.x) ** 2 + (route[0].y - start.y) ** 2)
        end_distance = np.sqrt((route[-1].x - goal.x) ** 2 + (route[-1].y - goal.y) ** 2)
        self.assertLess(start_distance, 0.3)  # Within reasonable distance
        self.assertLess(end_distance, 0.3)

    def test_find_route_nonexistent_map(self):
        """Test finding a route in a non-existent map raises KeyError."""
        start = Point(x=0.0, y=0.0, z=0.0)
        goal = Point(x=1.0, y=1.0, z=0.0)

        with self.assertRaises(KeyError):
            self.manager.find_route("nonexistent_map", start, goal)

    def test_find_route_no_path(self):
        """Test finding a route when no path exists returns empty list."""
        map_id = "test_map_blocked"

        # Create a map with two completely separate areas
        blocked_map = np.ones((20, 20), dtype=np.uint8) * 255
        blocked_map[8:12, :] = 0  # Add horizontal wall with NO gaps

        # Create graph
        self.manager.create_graph(
            map_id=map_id,
            image=blocked_map,
            resolution=self.test_resolution,
            safety_distance=self.test_safety_distance,
        )

        # Define start and goal points on opposite sides of the wall
        start = Point(x=0.25, y=0.25, z=0.0)  # Top side
        goal = Point(x=0.75, y=0.75, z=0.0)  # Bottom side

        # Find route - should return empty list since no path exists
        route = self.manager.find_route(map_id, start, goal)
        self.assertEqual(len(route), 0)


if __name__ == "__main__":
    unittest.main()
