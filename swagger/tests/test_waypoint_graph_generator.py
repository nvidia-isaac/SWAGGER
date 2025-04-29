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
from unittest.mock import patch

import cv2
import networkx as nx
import numpy as np

from swagger import WaypointGraphGenerator
from swagger.models import Point


class TestWaypointGraphGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = WaypointGraphGenerator()
        self.test_dir = tempfile.mkdtemp()

        # Create a simple test map (20x20 pixels)
        # With a corridor (white) through black obstacles
        self.test_map = np.zeros((20, 20), dtype=np.uint8)
        self.test_map[8:12, :] = 255  # Horizontal corridor
        self.test_map[:, 8:12] = 255  # Vertical corridor

        # Parameters for graph generation
        self.resolution = 0.05  # 5cm per pixel
        self.safety_distance = 0.1  # 10cm radius

        # Create a simple 100x100 map with a square obstacle
        map_size = 100
        self.simple_map = np.zeros((map_size, map_size), dtype=np.uint8)
        # Add a 20x20 square obstacle in the middle
        self.simple_map[40:60, 40:60] = 255

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_build_graph_from_grid_map(self):
        """Test basic graph generation from a simple map"""
        graph = self.generator.build_graph_from_grid_map(self.test_map, self.resolution, self.safety_distance)

        # Check graph structure
        self.assertIsNotNone(graph)
        self.assertGreater(len(graph.nodes), 0, "Graph should have nodes")
        self.assertGreater(len(graph.edges), 0, "Graph should have edges")

        # Check that nodes are properly converted to world coordinates
        for node, world_coords in graph.nodes(data="world"):
            self.assertIsInstance(node, int)
            self.assertGreaterEqual(world_coords[0], 0)
            self.assertGreaterEqual(world_coords[1], 0)
            self.assertEqual(world_coords[2], 0.0)

        # Check edge weights are positive
        self.assertTrue(all(w > 0 for _, _, w in graph.edges(data="weight")), "All edge weights should be positive")

        # Check that edge weights match actual distances between nodes
        for src, dst, weight in graph.edges(data="weight"):
            src_coords = graph.nodes[src]["world"]
            dst_coords = graph.nodes[dst]["world"]
            dist = np.sqrt((src_coords[0] - dst_coords[0]) ** 2 + (src_coords[1] - dst_coords[1]) ** 2)
            self.assertAlmostEqual(weight, dist, places=3)

    def test_empty_map(self):
        """Test graph generation with an empty map"""
        empty_map = np.zeros((10, 10), dtype=np.uint8)
        graph = self.generator.build_graph_from_grid_map(empty_map, self.resolution, self.safety_distance)

        # Should generate an empty graph
        self.assertEqual(len(graph.nodes), 0, "Empty map should have no nodes")
        self.assertEqual(len(graph.edges), 0, "Empty map should have no edges")

    def test_fully_free_map(self):
        """Test graph generation with a completely free map"""
        free_map = np.full((10, 10), 255, dtype=np.uint8)
        graph = self.generator.build_graph_from_grid_map(free_map, self.resolution, self.safety_distance)

        # Should generate a graph with some nodes
        self.assertGreater(len(graph.nodes), 0, "Free map should have nodes")
        # Add checks for grid-like structure
        self.assertGreater(len(graph.edges), 0, "Free map should have edges")

    def test_get_node_ids(self):
        """Test finding nearest nodes to query points"""
        # First build a graph
        self.generator.build_graph_from_grid_map(self.test_map, self.resolution, self.safety_distance)

        # Test points in the corridors
        test_points = [
            Point(x=0.5, y=0.5, z=0.0),  # Should be near corridor intersection
            Point(x=0.0, y=0.5, z=0.0),  # Should be in horizontal corridor
            Point(x=0.5, y=0.0, z=0.0),  # Should be in vertical corridor
        ]

        node_ids = self.generator.get_node_ids(test_points)

        # Check results
        self.assertEqual(len(node_ids), len(test_points), "Should return same number of IDs as input points")
        self.assertTrue(
            all(isinstance(idx, (int, type(None))) for idx in node_ids), "Results should be integers or None"
        )

    def test_get_node_ids_no_graph(self):
        """Test error handling when no graph exists"""
        with self.assertRaises(RuntimeError):
            self.generator.get_node_ids([Point(x=0.0, y=0.0, z=0.0)])

    def test_get_node_ids_invalid_points(self):
        """Test handling of points in obstacles"""
        # Build graph
        self.generator.build_graph_from_grid_map(self.test_map, self.resolution, self.safety_distance)

        # Test point in obstacle
        test_points = [Point(x=0.1, y=0.1, z=0.0)]  # Point in obstacle
        node_ids = self.generator.get_node_ids(test_points)

        self.assertEqual(len(node_ids), 1)
        self.assertIsNone(node_ids[0], "Point in obstacle should return None")

    def test_visualize_graph(self):
        """Test graph visualization"""
        # Build graph
        self.generator.build_graph_from_grid_map(self.test_map, self.resolution, self.safety_distance)

        # Create temporary directory for test outputs

        test_dir = tempfile.mkdtemp()
        try:
            # Test visualization
            self.generator.visualize_graph(
                output_dir=test_dir,
            )

            # Check that output files were created
            expected_files = ["waypoint_graph.png"]
            for filename in expected_files:
                filepath = os.path.join(test_dir, filename)
                self.assertTrue(os.path.exists(filepath), f"Expected output file {filename} not found")

                # Check that files are not empty
                self.assertGreater(os.path.getsize(filepath), 0, f"Output file {filename} is empty")

        finally:
            # Clean up temporary directory
            shutil.rmtree(test_dir)

    def test_build_graph_with_threshold(self):
        """Test building graph with different occupancy thresholds."""
        # Create test map with three intensity regions
        test_map = np.ones((30, 30), dtype=np.uint8) * 127  # Middle intensity
        test_map[5:25, 5:25] = 200  # Large free area with high threshold
        test_map[10:20, 10:20] = 50  # Obstacle area with low threshold

        generator = WaypointGraphGenerator()

        # Test with high threshold (more obstacles)
        high_graph = generator.build_graph_from_grid_map(
            image=test_map,
            resolution=0.05,
            safety_distance=0.2,
            occupancy_threshold=210,  # High threshold means more cells are obstacles
        )

        # Test with low threshold (fewer obstacles)
        low_graph = generator.build_graph_from_grid_map(
            image=test_map,
            resolution=0.05,
            safety_distance=0.2,
            occupancy_threshold=30,  # Low threshold means fewer cells are obstacles
        )

        # Low threshold should result in more nodes (fewer obstacles)
        self.assertGreater(len(low_graph.nodes), len(high_graph.nodes))
        self.assertEqual(generator._occupancy_threshold, 30)

    def test_inflate_obstacles_with_threshold(self):
        """Test obstacle inflation with different thresholds."""
        # Create test map with gradient intensities
        test_map = np.linspace(0, 255, 100).reshape(10, 10).astype(np.uint8)

        generator = WaypointGraphGenerator()

        # Test with low threshold
        generator._occupancy_threshold = 100
        generator._resolution = 0.05
        generator._safety_distance = 0.2
        generator._original_map = test_map
        generator._distance_transform()
        inflated_low = generator._inflated_map.copy()

        # Test with high threshold
        generator._occupancy_threshold = 200
        generator._distance_transform()
        inflated_high = generator._inflated_map.copy()

        # High threshold should result in more occupied cells
        self.assertGreater(np.sum(inflated_high), np.sum(inflated_low))

    def test_visualize_graph_with_threshold(self):
        """Test graph visualization with different thresholds."""
        # Create test map with three intensity regions
        test_map = np.ones((20, 20), dtype=np.uint8) * 127
        test_map[5:15, :] = 200
        test_map[10:20, :] = 50

        generator = WaypointGraphGenerator()

        # Build graph with specific threshold
        threshold = 150
        generator.build_graph_from_grid_map(
            image=test_map, resolution=0.05, safety_distance=0.2, occupancy_threshold=threshold
        )

        # Create visualization
        vis_path = os.path.join(self.test_dir, "vis_test.png")
        generator.visualize_graph(
            output_dir=self.test_dir,
            output_filename="vis_test.png",
        )

        # Load and check visualization
        vis_img = cv2.imread(vis_path)
        self.assertIsNotNone(vis_img)

        # Check that visualization uses the same threshold
        # White pixels should correspond to values > threshold
        expected_white = np.sum(test_map > threshold)
        actual_white = np.sum(np.all(vis_img == [255, 255, 255], axis=2))
        self.assertAlmostEqual(expected_white, actual_white, delta=100)  # Allow some difference due to graph overlay

    def test_coordinate_transforms(self):
        """Test pixel to world and world to pixel coordinate transforms."""
        generator = WaypointGraphGenerator()

        # Set up transform parameters
        resolution = 0.05  # 5cm per pixel
        x_offset = 1.0  # 1m offset in x
        y_offset = 2.0  # 2m offset in y
        rotation = np.pi / 4  # 45 degree rotation

        # Initialize generator with transform
        generator.build_graph_from_grid_map(
            image=np.zeros((10, 10)),  # Dummy image
            resolution=resolution,
            safety_distance=0.1,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )

        # Test pixel to world conversion
        pixel_y, pixel_x = 10.0, 20.0
        world_point = generator._pixel_to_world(pixel_y, pixel_x)

        # Test world to pixel conversion
        pixel_y_back, pixel_x_back = generator._world_to_pixel(world_point)

        # Verify round trip conversion
        self.assertAlmostEqual(pixel_y, pixel_y_back, places=5)
        self.assertAlmostEqual(pixel_x, pixel_x_back, places=5)

        # Test specific transform values
        origin_world = generator._pixel_to_world(0, 0)
        self.assertAlmostEqual(origin_world.x, x_offset, places=5)
        self.assertAlmostEqual(origin_world.y, y_offset + 10 * resolution, places=5)
        self.assertAlmostEqual(origin_world.z, 0.0, places=5)

    def test_graph_with_transform(self):
        """Test graph generation with transform applied."""
        # Create test map
        test_map = np.zeros((20, 20), dtype=np.uint8)
        test_map[8:12, :] = 255  # Horizontal corridor

        # Set up transform
        x_offset = 1.0
        y_offset = 2.0
        rotation = np.pi / 4  # 45 degrees

        generator = WaypointGraphGenerator()
        graph = generator.build_graph_from_grid_map(
            image=test_map,
            resolution=0.05,
            safety_distance=0.1,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )
        # Verify nodes are transformed
        for _, data in graph.nodes(data=True):
            world_coords = data["world"]
            pixel_coords = data["pixel"]
            # All nodes should be offset by at least the translation
            dist = np.sqrt(
                (world_coords[0] - pixel_coords[0] * 0.05) ** 2 + (world_coords[1] - pixel_coords[1] * 0.05) ** 2
            )
            self.assertGreater(dist, 0.0)

    def test_transform_origin(self):
        """Test that transform correctly handles the origin."""
        generator = WaypointGraphGenerator()

        # Set up transform
        resolution = 0.05
        x_offset = 1.0
        y_offset = 2.0
        rotation = np.pi / 4  # 45 degrees

        generator.build_graph_from_grid_map(
            image=np.zeros((10, 10)),
            resolution=resolution,
            safety_distance=0.1,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )

        # Origin in pixel coordinates should map to offset in world coordinates
        world_origin = generator._pixel_to_world(0, 0)
        self.assertAlmostEqual(world_origin.x, x_offset, places=5)
        self.assertAlmostEqual(world_origin.y, y_offset + 10 * resolution, places=5)
        self.assertAlmostEqual(world_origin.z, 0.0, places=5)

    def test_find_obstacle_contours(self):
        """Test contour finding on a simple map."""
        # Setup generator
        self.generator._safety_distance = 0.1
        self.generator._resolution = 0.05
        self.generator._original_map = self.simple_map
        self.generator._distance_transform()

        # Find contours
        contours = self.generator._find_obstacle_contours(
            1.5 * self.generator._safety_distance / self.generator._resolution
        )

        # Should find one contour for the square
        self.assertEqual(len(contours), 1)

        # Square contour should have 4 points (in CV2's compressed format)
        self.assertGreaterEqual(len(contours[0]), 4)

    def test_is_valid_point(self):
        """Test point validation."""
        self.generator._inflated_map = self.simple_map

        # Test points in free space
        self.assertTrue(self.generator._is_valid_point(0, 0))  # Corner
        self.assertTrue(self.generator._is_valid_point(20, 20))  # Free space

        # Test points in obstacle
        self.assertFalse(self.generator._is_valid_point(50, 50))  # Inside obstacle

        # Test out of bounds
        self.assertFalse(self.generator._is_valid_point(-1, 0))
        self.assertFalse(self.generator._is_valid_point(0, -1))
        self.assertFalse(self.generator._is_valid_point(100, 0))
        self.assertFalse(self.generator._is_valid_point(0, 100))

    def test_sample_obstacle_boundaries(self):
        """Test full boundary sampling process."""
        # Setup generator
        self.generator._safety_distance = 0.3
        self.generator._resolution = 0.05
        self.generator._original_map = ~self.simple_map
        self.generator._distance_transform()

        # Create empty graph
        G = nx.Graph()

        # Sample boundaries
        self.generator._sample_obstacle_boundaries(G, sample_distance=10)

        # Should have found some nodes
        self.assertGreater(len(G.nodes()), 0)

        # All nodes should be in free space
        for node, data in G.nodes(data=True):
            y, x = node
            self.assertFalse(self.simple_map[y, x])
            node_type = data.get("node_type", "default")
            self.assertEqual(node_type, "boundary")

        # Should have some edges
        self.assertGreater(len(G.edges()), 0)

        # All edges should be marked as contour type
        for _, _, data in G.edges(data=True):
            self.assertEqual(data["edge_type"], "contour")

        # Edges should form a cycle (for this simple square obstacle)
        cycles = nx.cycle_basis(G)
        self.assertGreater(len(cycles), 0)

    def test_connect_contour_nodes(self):
        """Test connecting contour nodes."""
        # Create a simple square of nodes
        contour_nodes = [(0, 0), (0, 10), (10, 10), (10, 0)]
        G = nx.Graph()
        for node in contour_nodes:
            G.add_node(node)

        # Setup generator
        self.generator._inflated_map = np.zeros((20, 20), dtype=np.uint8)
        self.generator._resolution = 0.05

        # Connect nodes
        self.generator._connect_contour_nodes(contour_nodes, G)

        # Should have 4 edges forming a square
        self.assertEqual(len(G.edges()), 4)

        # All edges should be contour type
        for _, _, data in G.edges(data=True):
            self.assertEqual(data["edge_type"], "contour")

    def test_visualize_graph_without_building(self):
        """Test visualizing the graph before it is built."""
        # Attempt to visualize without building a graph
        with self.assertRaises(RuntimeError):
            self.generator.visualize_graph(
                output_dir=self.test_dir,
                output_filename="unbuilt_graph.png",
            )

    def test_get_node_ids_with_transform(self):
        """Test finding nearest nodes with a non-identity transform."""
        # Create a simple test map with a cross shape
        test_map = np.zeros((20, 20), dtype=np.uint8)
        test_map[8:12, :] = 255  # Horizontal corridor
        test_map[:, 8:12] = 255  # Vertical corridor

        # Set up transform parameters
        resolution = 0.05  # 5cm per pixel
        safety_distance = 0.1  # 10cm
        x_offset = 1.0  # 1m offset in x
        y_offset = 2.0  # 2m offset in y
        rotation = np.pi / 4  # 45 degree rotation

        # Build graph with transform
        graph = self.generator.build_graph_from_grid_map(
            image=test_map,
            resolution=resolution,
            safety_distance=safety_distance,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )

        # Create test points by transforming known free pixels to world coordinates
        # Center of horizontal corridor
        pixel_h = (10, 0)  # y, x coordinates in pixel space
        world_h = self.generator._pixel_to_world(pixel_h[0], pixel_h[1])

        # Center of vertical corridor
        pixel_v = (0, 10)  # y, x coordinates in pixel space
        world_v = self.generator._pixel_to_world(pixel_v[0], pixel_v[1])

        # Point in obstacle
        pixel_obs = (0, 0)  # y, x coordinates in pixel space
        world_obs = self.generator._pixel_to_world(pixel_obs[0], pixel_obs[1])

        test_points = [
            world_h,  # Point in horizontal corridor
            world_v,  # Point in vertical corridor
            world_obs,  # Point in obstacle
        ]

        # Get nearest nodes
        node_ids = self.generator.get_node_ids(test_points)

        # Verify results
        self.assertEqual(len(node_ids), len(test_points))
        self.assertIsNotNone(node_ids[0], "Node at horizontal corridor center should be found")
        self.assertIsNotNone(node_ids[1], "Node at vertical corridor center should be found")
        self.assertIsNone(node_ids[2], "Point in obstacle should return None")

        # Verify that found nodes are actually close to the query points
        for point, node_id in zip(test_points[:2], node_ids[:2]):  # Check first two points (valid ones)
            if node_id is not None:
                world_coords = graph.nodes[node_id]["world"]

                # Calculate distance in world coordinates
                dist = np.sqrt((point.x - world_coords[0]) ** 2 + (point.y - world_coords[1]) ** 2)

                # Distance should be reasonable given map size and resolution
                self.assertLess(dist, 0.5, f"Node too far from query point. Distance: {dist}m")

    def test_get_node_ids_rotation_invariance(self):
        """Test that node finding works similarly for different rotations."""
        # Create a simple cross-shaped map
        test_map = np.zeros((20, 20), dtype=np.uint8)
        test_map[8:12, :] = 255  # Horizontal corridor
        test_map[:, 8:12] = 255  # Vertical corridor

        resolution = 0.05
        safety_distance = 0.1
        x_offset = 1.0
        y_offset = 2.0

        # Test with different rotations
        rotations = [0, np.pi / 4, np.pi / 2, np.pi]
        results = []

        for rotation in rotations:
            # Build graph with current rotation
            self.generator.build_graph_from_grid_map(
                image=test_map,
                resolution=resolution,
                safety_distance=safety_distance,
                x_offset=x_offset,
                y_offset=y_offset,
                rotation=rotation,
            )

            # Create test point by transforming a known free pixel
            # Use center of horizontal corridor
            pixel_point = (10, 10)  # Center of cross in pixel coordinates
            test_point = self.generator._pixel_to_world(pixel_point[0], pixel_point[1])

            # Get nearest node
            node_id = self.generator.get_node_ids([test_point])[0]
            results.append(node_id is not None)

        # Point should be findable with any rotation
        self.assertTrue(all(results), "Point should be findable at all rotations")

    def test_cuda_runtime_error(self):
        """Test that the library can handle CUDA runtime errors."""
        # Create a simple cross-shaped map
        test_map = np.zeros((20, 20), dtype=np.uint8)
        test_map[8:12, :] = 255  # Horizontal corridor
        test_map[:, 8:12] = 255  # Vertical corridor

        with patch("swagger.waypoint_graph_generator.thin", side_effect=RuntimeError("Test error")):
            graph = self.generator.build_graph_from_grid_map(image=test_map, resolution=0.05, safety_distance=0.1)
            self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
