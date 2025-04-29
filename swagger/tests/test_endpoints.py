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

import cv2
import numpy as np
from fastapi.testclient import TestClient

from swagger.endpoints import app
from swagger.graph_manager import GraphManager


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.vis_dir = os.path.join(self.test_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        # Create test client with visualization directory
        app.dependency_overrides = {}  # Reset any previous overrides
        global graph_manager
        graph_manager = GraphManager(visualization_dir=self.vis_dir)
        self.client = TestClient(app)

        # Create a simple test map
        self.test_map = np.ones((20, 20), dtype=np.uint8) * 255
        self.test_map[8:12, :] = 0  # Add horizontal wall

        # Save test map as PNG
        self.test_map_path = os.path.join(self.test_dir, "test_map.png")
        cv2.imwrite(self.test_map_path, self.test_map)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            if os.path.isdir(file_path):
                for subfile in os.listdir(file_path):
                    os.remove(os.path.join(file_path, subfile))
                os.rmdir(file_path)
            else:
                os.remove(file_path)
        os.rmdir(self.test_dir)

    def test_health(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("errors", data)

    def test_post_graph(self):
        """Test graph generation endpoint."""
        map_data = {
            "map_id": "test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "rotation": 0.0,
        }

        # Test with file upload
        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)

        self.assertEqual(response.status_code, 201)
        graph = response.json()
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        self.assertIn("weights", graph)
        self.assertIn("offsets", graph)

    def test_post_graph_empty_map_id(self):
        """Test graph generation with empty map ID."""
        map_data = {
            "map_id": "",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
        }  # Empty map ID

        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

        # Test with whitespace-only map ID
        map_data["map_id"] = "   "
        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_get_graph(self):
        """Test retrieving a graph."""
        map_data = {
            "map_id": "test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "rotation": 0.0,
        }

        # Create graph
        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post(
                "/graph/generate", data=map_data, files=files  # FastAPI will handle converting this to form data
            )
        self.assertEqual(response.status_code, 201)  # Verify creation succeeded

        # Now retrieve it
        response = self.client.get("/graph?map_id=test_map")
        self.assertEqual(response.status_code, 200)
        graph = response.json()
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)

    def test_get_graph_empty_map_id(self):
        """Test retrieving graph with empty map ID."""
        response = self.client.get("/graph?map_id=")
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

        response = self.client.get("/graph?map_id=   ")
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_get_nonexistent_graph(self):
        """Test retrieving non-existent graph returns 404."""
        response = self.client.get("/graph?map_id=nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_get_nearest_nodes(self):
        """Test finding nearest nodes."""
        map_data = {
            "map_id": "test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "rotation": 0.0,
        }

        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)
        self.assertEqual(response.status_code, 201)  # Verify creation succeeded

        # Test points
        points = {"points": [{"x": 0.25, "y": 0.25, "z": 0.0}, {"x": 0.25, "y": 0.25, "z": 0.0}]}

        response = self.client.post("/graph/nearest_nodes?map_id=test_map", json=points)
        self.assertEqual(response.status_code, 200)
        node_indices = response.json()
        self.assertEqual(len(node_indices), 2)
        self.assertIsNotNone(node_indices[0])
        self.assertEqual(node_indices[0], node_indices[1])

    def test_get_nearest_nodes_nonexistent_map(self):
        """Test finding nearest nodes for non-existent map returns 404."""
        points = {"points": [{"x": 0.0, "y": 0.0, "z": 0.0}]}
        response = self.client.post("/graph/nearest_nodes?map_id=nonexistent", json=points)
        self.assertEqual(response.status_code, 404)

    def test_visualize_graph(self):
        """Test graph visualization."""
        map_data = {
            "map_id": "test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "rotation": 0.0,
        }

        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)
        self.assertEqual(response.status_code, 201)  # Verify creation succeeded

        # Verify graph exists
        response = self.client.get("/graph?map_id=test_map")
        self.assertEqual(response.status_code, 200)

        # Get visualization
        response = self.client.get("/graph/visualize?map_id=test_map")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_visualize_nonexistent_graph(self):
        """Test visualizing non-existent graph returns 404."""
        response = self.client.get("/graph/visualize?map_id=nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_get_route(self):
        """Test finding a route between two points."""
        # Create a simple map with two clear areas separated by a wall with a gap
        route_test_map = np.ones((20, 20), dtype=np.uint8) * 255
        route_test_map[8:12, :] = 0  # Add horizontal wall
        route_test_map[8:12, 5:15] = 255  # Add a gap in the wall

        # Save test map
        route_map_path = os.path.join(self.test_dir, "route_map.png")
        cv2.imwrite(route_map_path, route_test_map)

        # Create graph
        map_data = {
            "map_id": "route_test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
        }

        with open(route_map_path, "rb") as f:
            files = {"map_file": ("route_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)

        self.assertEqual(response.status_code, 201)  # Verify creation succeeded

        # Define start and goal points on opposite sides of the wall
        request_data = {
            "start": {"x": 0.25, "y": 0.25, "z": 0.0},  # Top side
            "goal": {"x": 0.25, "y": 0.75, "z": 0.0},  # Bottom side
        }

        # Query for a route using POST with JSON body
        response = self.client.post("/graph/route?map_id=route_test_map", json=request_data)

        self.assertEqual(response.status_code, 200)
        route = response.json()

        # Verify route exists and has reasonable properties
        self.assertIsInstance(route, list)
        self.assertGreater(len(route), 0)

        # Verify each point in route has x, y, z coordinates
        for point in route:
            self.assertIn("x", point)
            self.assertIn("y", point)
            self.assertIn("z", point)

        # Verify start and end points are close to requested points
        start = request_data["start"]
        goal = request_data["goal"]
        start_distance = ((route[0]["x"] - start["x"]) ** 2 + (route[0]["y"] - start["y"]) ** 2) ** 0.5
        end_distance = ((route[-1]["x"] - goal["x"]) ** 2 + (route[-1]["y"] - goal["y"]) ** 2) ** 0.5
        self.assertLess(start_distance, 0.3)  # Within reasonable distance
        self.assertLess(end_distance, 0.3)

    def test_get_route_nonexistent_map(self):
        """Test finding a route for non-existent map returns 404."""
        # Create the request body with start and goal points
        request_data = {"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "goal": {"x": 1.0, "y": 1.0, "z": 0.0}}

        # Use POST request with JSON body
        response = self.client.post("/graph/route?map_id=nonexistent", json=request_data)

        self.assertEqual(response.status_code, 404)

    def test_get_route_no_path(self):
        """Test finding a route when no path exists returns empty list."""
        # Create a map with two completely separate areas
        blocked_map = np.ones((20, 20), dtype=np.uint8) * 255
        blocked_map[8:12, :] = 0  # Add horizontal wall with NO gaps

        # Save test map
        blocked_map_path = os.path.join(self.test_dir, "blocked_map.png")
        cv2.imwrite(blocked_map_path, blocked_map)

        # Create graph
        map_data = {
            "map_id": "blocked_test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 127,
        }

        with open(blocked_map_path, "rb") as f:
            files = {"map_file": ("blocked_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)

        self.assertEqual(response.status_code, 201)  # Verify creation succeeded

        # Define start and goal points on opposite sides of the wall
        request_data = {
            "start": {"x": 0.25, "y": 0.25, "z": 0.0},  # Top side
            "goal": {"x": 0.75, "y": 0.75, "z": 0.0},  # Bottom side
        }

        # Query for a route using POST with JSON body
        response = self.client.post("/graph/route?map_id=blocked_test_map", json=request_data)

        self.assertEqual(response.status_code, 200)
        route = response.json()

        # Verify route is empty since no path exists
        self.assertEqual(len(route), 0)

    def test_post_graph_with_threshold(self):
        """Test graph generation with different occupancy thresholds."""
        # Create test map with three intensity regions
        test_map = np.ones((30, 30), dtype=np.uint8) * 127  # Middle intensity
        test_map[5:25, 5:25] = 200  # Large free area with high threshold
        test_map[10:20, 10:20] = 50  # Obstacle area with low threshold

        # Save test map
        threshold_map_path = os.path.join(self.test_dir, "threshold_map.png")
        cv2.imwrite(threshold_map_path, test_map)

        # Test with high threshold (more obstacles)
        map_data_high = {
            "map_id": "test_map_high",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 210,  # High threshold means more cells are obstacles
        }

        with open(threshold_map_path, "rb") as f:
            files = {"map_file": ("threshold_map.png", f, "image/png")}
            response_high = self.client.post("/graph/generate", data=map_data_high, files=files)

        # Test with low threshold (fewer obstacles)
        map_data_low = {
            "map_id": "test_map_low",
            "resolution": 0.05,
            "safety_distance": 0.2,
            "occupancy_threshold": 30,  # Low threshold means fewer cells are obstacles
        }

        with open(threshold_map_path, "rb") as f:
            files = {"map_file": ("threshold_map.png", f, "image/png")}
            response_low = self.client.post("/graph/generate", data=map_data_low, files=files)

        self.assertEqual(response_high.status_code, 201)
        self.assertEqual(response_low.status_code, 201)

        # Low threshold should result in more nodes (fewer obstacles)
        graph_high = response_high.json()
        graph_low = response_low.json()
        self.assertGreater(len(graph_low["nodes"]), len(graph_high["nodes"]))

    def test_post_graph_missing_threshold(self):
        """Test graph generation without occupancy threshold."""
        map_data = {
            "map_id": "test_map",
            "resolution": 0.05,
            "safety_distance": 0.2,
            # Missing occupancy_threshold
        }

        with open(self.test_map_path, "rb") as f:
            files = {"map_file": ("test_map.png", f, "image/png")}
            response = self.client.post("/graph/generate", data=map_data, files=files)

        # Should fail because occupancy_threshold does not a default value
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
