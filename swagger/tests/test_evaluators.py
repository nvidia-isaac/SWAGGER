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

import time
import unittest
from unittest.mock import Mock

import networkx as nx
import numpy as np

from swagger.graph_evaluator import WaypointGraphEvaluator
from swagger.performance_evaluator import PerformanceEvaluator, PerformanceResult


class TestWaypointGraphEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.graph = nx.Graph()
        pixels = [(1, 1), (5, 1), (5, 5), (1, 5)]
        world_coords = [(0.05, 0.05), (0.25, 0.05), (0.25, 0.25), (0.05, 0.25)]
        for i, (pixel, world_coord) in enumerate(zip(pixels, world_coords)):
            self.graph.add_node(i, world=world_coord, pixel=pixel)
        self.graph.add_edge(0, 1, weight=0.2)
        self.graph.add_edge(1, 2, weight=0.2)
        self.graph.add_edge(2, 3, weight=0.2)
        self.graph.add_edge(3, 0, weight=0.2)

        # Create a simple occupancy map (8x7)
        # 255 = free space, 0 = obstacle
        self.occupancy_map = np.array(
            [
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 0, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        )

        self.resolution = 0.05
        self.safety_distance = 0.01
        self.occupancy_threshold = 127
        self.evaluator = WaypointGraphEvaluator(
            self.graph, self.occupancy_map, self.resolution, self.safety_distance, self.occupancy_threshold
        )

    def test_coverage_metrics(self):
        """Test calculation of coverage metrics."""
        metrics = self.evaluator.calculate_coverage_metrics()

        self.assertIn("free_space_coverage", metrics)
        self.assertIn("average_distance_to_node", metrics)
        self.assertIn("max_distance_to_node", metrics)
        self.assertIn("coverage_efficiency", metrics)

        # Check value ranges
        self.assertGreaterEqual(metrics["free_space_coverage"], 0.0)
        self.assertLessEqual(metrics["free_space_coverage"], 1.0)
        self.assertGreaterEqual(metrics["average_distance_to_node"], 0.0)
        self.assertGreaterEqual(metrics["max_distance_to_node"], 0.0)

    def test_graph_metrics(self):
        """Test calculation of graph structure metrics."""
        metrics = self.evaluator.calculate_graph_metrics()

        self.assertEqual(metrics["num_nodes"], 4)
        self.assertEqual(metrics["num_edges"], 4)
        self.assertAlmostEqual(metrics["average_degree"], 2.0)
        self.assertAlmostEqual(metrics["average_edge_length"], 0.2)

    def test_path_metrics(self):
        """Test calculation of path planning metrics."""
        metrics = self.evaluator.calculate_path_metrics(num_samples=1)

        self.assertIn("average_path_length", metrics)
        self.assertIn("average_path_smoothness", metrics)
        self.assertIn("path_success_rate", metrics)
        self.assertIn("average_path_clearance", metrics)

        # Check value ranges
        self.assertGreaterEqual(metrics["path_success_rate"], 0.0)
        self.assertLessEqual(metrics["path_success_rate"], 1.0)
        self.assertGreaterEqual(metrics["average_path_smoothness"], 0.0)
        self.assertLessEqual(metrics["average_path_smoothness"], 1.0)

    def test_validation_metrics(self):
        """Test calculation of validation metrics."""
        metrics = self.evaluator.calculate_validation_metrics()

        self.assertIn("collision_free", metrics)
        self.assertIn("node_validity", metrics)
        self.assertIn("connectivity", metrics)
        self.assertIn("edge_length_validity", metrics)

        # Check graph properties
        self.assertTrue(metrics["connectivity"])  # Graph should be connected
        self.assertTrue(metrics["node_validity"])  # All nodes should be valid
        self.assertTrue(metrics["edge_length_validity"])  # All edges should be valid length

    def test_empty_graph(self):
        """Test evaluator with empty graph."""
        empty_graph = nx.Graph()
        evaluator = WaypointGraphEvaluator(
            empty_graph, self.occupancy_map, self.resolution, self.safety_distance, self.occupancy_threshold
        )

        metrics = evaluator.evaluate_all(num_path_samples=10)

        # Check that metrics handle empty graph gracefully
        self.assertEqual(metrics["graph_structure"]["num_nodes"], 0)
        self.assertEqual(metrics["graph_structure"]["num_edges"], 0)
        self.assertEqual(metrics["path_planning"]["path_success_rate"], 0.0)


class TestPerformanceEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = PerformanceEvaluator()

    def test_timing_measurement(self):
        """Test basic timing functionality."""
        self.evaluator.start("test_section")
        time.sleep(0.1)  # Simulate some work
        duration = self.evaluator.stop("test_section")

        self.assertGreaterEqual(duration, 0.1)
        self.assertLess(duration, 0.2)  # Allow some overhead

    def test_memory_measurement(self):
        """Test memory usage measurement."""
        initial_memory = self.evaluator.get_peak_memory()

        # Allocate some memory
        large_array = np.random.randint(0, 255, (1000, 1000))  # noqa: F841

        peak_memory = self.evaluator.get_peak_memory()
        self.assertGreater(peak_memory, initial_memory)

    def test_query_performance(self):
        """Test query performance evaluation."""
        # Create mock graph generator
        mock_generator = Mock()
        mock_generator._original_map = np.zeros((100, 100))
        mock_generator._resolution = 0.05
        mock_generator.get_node_ids = Mock(return_value=[0])

        metrics = self.evaluator.evaluate_query_performance(mock_generator, num_queries=10)

        self.assertIn("average_query_time", metrics)
        self.assertIn("max_query_time", metrics)
        self.assertIn("min_query_time", metrics)
        self.assertIn("total_query_time", metrics)

        # Check that mock was called correct number of times
        self.assertEqual(mock_generator.get_node_ids.call_count, 10)

    def test_get_results(self):
        """Test getting final performance results."""
        self.evaluator.start("total")
        self.evaluator.start("section1")
        large_array = np.random.randint(0, 255, (1000, 1000))  # noqa: F841
        time.sleep(0.1)
        self.evaluator.stop("section1")
        self.evaluator.stop("total")

        results = self.evaluator.get_results()

        self.assertIsInstance(results, PerformanceResult)
        self.assertGreater(results.total_time, 0.0)
        self.assertGreater(results.peak_memory, 0.0)
        self.assertIn("section1", results.breakdown)

    def test_multiple_sections(self):
        """Test timing multiple sections."""
        self.evaluator.start("total")

        self.evaluator.start("section1")
        time.sleep(0.1)
        self.evaluator.stop("section1")

        self.evaluator.start("section2")
        time.sleep(0.2)
        self.evaluator.stop("section2")

        self.evaluator.stop("total")

        results = self.evaluator.get_results()

        self.assertIn("section1", results.breakdown)
        self.assertIn("section2", results.breakdown)
        self.assertGreater(results.breakdown["section2"], results.breakdown["section1"])

    def test_invalid_section(self):
        """Test stopping an invalid section."""
        duration = self.evaluator.stop("nonexistent")
        self.assertEqual(duration, 0.0)


if __name__ == "__main__":
    unittest.main()
