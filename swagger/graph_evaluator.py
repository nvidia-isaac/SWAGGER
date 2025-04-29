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
import networkx as nx
import numpy as np

from swagger.logger import Logger


class WaypointGraphEvaluator:
    """Class to evaluate waypoint graph quality using various metrics."""

    def __init__(
        self,
        graph: nx.Graph,
        occupancy_map: np.ndarray,
        resolution: float,
        safety_distance: float,
        occupancy_threshold: int,
    ):
        """
        Initialize the evaluator.

        Args:
            graph: NetworkX graph to evaluate
            occupancy_map: Binary numpy array where True indicates obstacles
            resolution: Map resolution in meters/pixel
        """
        self._logger = Logger(__name__)
        if not self._validate_graph(graph, occupancy_map):
            raise ValueError("Graph is out of bounds of the occupancy map")

        self._graph = graph
        self._occupancy_map = occupancy_map
        self._resolution = resolution
        self._safety_distance = safety_distance
        self._occupancy_threshold = occupancy_threshold
        # Precompute the distance transform of the occupancy map
        self._distance_map = cv2.distanceTransform(
            (self._occupancy_map > self._occupancy_threshold).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        self._inflated_map = (self._distance_map < self._safety_distance / self._resolution).astype(np.uint8)

    def _validate_graph(self, graph: nx.Graph, occupancy_map: np.ndarray) -> bool:
        """Validate the graph is within the bounds of the occupancy map."""
        for _, pixel in graph.nodes(data="pixel"):
            y, x = pixel
            if y < 0 or y >= occupancy_map.shape[0] or x < 0 or x >= occupancy_map.shape[1]:
                self._logger.error(
                    f"Node {y}, {x} is out of bounds of the occupancy map (shape: {occupancy_map.shape})"
                )
                return False
        return True

    def evaluate_all(self, num_path_samples: int = 1000, print_metrics: bool = False) -> dict[str, dict[str, float]]:
        """
        Calculate all evaluation metrics.

        Args:
            num_path_samples: Number of random paths to sample for path metrics

        Returns:
            Dictionary containing all metric categories and their values
        """
        metrics = {
            "coverage": self.calculate_coverage_metrics(),
            "graph_structure": self.calculate_graph_metrics(),
            "path_planning": self.calculate_path_metrics(num_path_samples),
            "validation": self.calculate_validation_metrics(),
        }
        if print_metrics:
            self._print_metrics(metrics)
        return metrics

    def calculate_coverage_metrics(self) -> dict[str, float]:
        """
        Calculate metrics related to graph coverage of free space.

        Returns:
            Dictionary containing:
            - free_space_coverage: % of free space with at least one node in the region
            - average_distance_to_node: Average distance from free cells to nearest node
            - max_distance_to_node: Maximum distance from any free cell to nearest node
            - coverage_efficiency: Ratio of coverage to number of nodes
        """
        self._logger.info("Calculating coverage metrics...")

        # Create binary free space mask
        free_space_mask = (self._occupancy_map > self._occupancy_threshold).astype(np.uint8)

        if np.sum(free_space_mask) == 0 or len(self._graph) == 0:
            self._logger.warning("No free space or no nodes in graph")
            return {
                "free_space_coverage": 0.0,
                "average_distance_to_node": np.nan,
                "max_distance_to_node": np.nan,
                "coverage_efficiency": 0.0,
            }

        # Segment free space into connected regions
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(free_space_mask, connectivity=8)

        # Create a binary image with only nodes marked
        node_image = np.zeros_like(self._occupancy_map, dtype=np.uint8)
        for _, pixel in self._graph.nodes(data="pixel"):
            y, x = pixel
            if 0 <= y < node_image.shape[0] and 0 <= x < node_image.shape[1]:
                node_image[y, x] = 255

        # Compute distance transform from nodes
        dist_transform = cv2.distanceTransform(255 - node_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Initialize metrics
        total_free_pixels = 0
        covered_free_pixels = 0
        all_distances = []

        # Analyze each connected region separately (skip label 0 which is background)
        for label in range(1, num_labels):
            # Create mask for this region
            region_mask = labels == label
            region_size = np.sum(region_mask)

            # Get distances for this region
            region_distances = dist_transform[region_mask]

            # Check if region has any nodes
            has_nodes = np.any(node_image[region_mask] > 0)

            # Update overall metrics
            total_free_pixels += region_size
            if has_nodes:
                covered_free_pixels += region_size
                all_distances.extend(region_distances)

        # Calculate overall metrics
        overall_coverage = covered_free_pixels / total_free_pixels if total_free_pixels > 0 else 0
        avg_distance = np.mean(all_distances) if all_distances else np.nan
        max_distance = np.max(all_distances) if all_distances else np.nan
        coverage_efficiency = overall_coverage / len(self._graph) if len(self._graph) > 0 else 0

        return {
            "free_space_coverage": float(overall_coverage),
            "average_distance_to_node": float(avg_distance),
            "max_distance_to_node": float(max_distance),
            "coverage_efficiency": float(coverage_efficiency),
        }

    def calculate_graph_metrics(self) -> dict[str, float]:
        """
        Calculate metrics about graph structure and connectivity.

        Returns:
            Dictionary containing:
            - num_nodes: Total number of nodes
            - num_edges: Total number of edges
            - average_degree: Average node degree
            - average_edge_length: Average edge length in pixels
            - graph_diameter: Longest shortest path
            - average_clustering: Average clustering coefficient
        """
        self._logger.info("Calculating graph structure metrics...")

        if len(self._graph) == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "average_degree": 0.0,
                "average_edge_length": 0.0,
                "graph_diameter": 0,
                "average_clustering": 0.0,
            }

        # Basic graph metrics
        metrics = {
            "num_nodes": len(self._graph.nodes()),
            "num_edges": len(self._graph.edges()),
            "average_degree": float(np.mean([d for _, d in self._graph.degree()])),
            "average_edge_length": float(np.mean([d["weight"] for _, _, d in self._graph.edges(data=True)])),
        }

        # More complex metrics that might fail
        try:
            metrics["graph_diameter"] = nx.diameter(self._graph)
        except nx.NetworkXError:
            metrics["graph_diameter"] = float("inf")

        metrics["average_clustering"] = float(nx.average_clustering(self._graph))

        return metrics

    def calculate_path_metrics(self, num_samples: int = 1000) -> dict[str, float]:
        """
        Calculate metrics related to path planning quality.

        Args:
            num_samples: Number of random paths to sample

        Returns:
            Dictionary containing:
            - average_path_length: Average length of shortest paths
            - average_path_smoothness: Average path smoothness
            - path_success_rate: % of valid paths found between random points
            - average_path_clearance: Average distance to obstacles along paths
        """
        self._logger.info("Calculating path planning metrics...")

        nodes = list(self._graph.nodes())
        if len(nodes) < 2:
            return {
                "average_path_length": 0.0,
                "average_path_smoothness": 0.0,
                "path_success_rate": 0.0,
                "average_path_clearance": 0.0,
            }

        # Sample random node pairs and find paths
        num_paths = 0
        path_lengths = []
        path_smoothness = []
        path_clearances = []

        import random

        for _ in range(num_samples):
            start, end = random.sample(nodes, 2)
            try:
                path = nx.shortest_path(self._graph, start, end, weight="weight")
                num_paths += 1

                # Calculate path length
                length = sum(self._graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
                path_lengths.append(length)

                pixel_path = [self._graph.nodes[node]["pixel"] for node in path]
                # Calculate path smoothness (using angle between consecutive edges)
                smoothness = self._calculate_path_smoothness(pixel_path)
                path_smoothness.append(smoothness)

                # Calculate average clearance along path
                clearance = self._calculate_path_clearance(pixel_path)
                path_clearances.append(clearance)

            except nx.NetworkXNoPath:
                continue

        if num_paths == 0:
            return {
                "average_path_length": 0.0,
                "average_path_smoothness": 0.0,
                "path_success_rate": 0.0,
                "average_path_clearance": 0.0,
            }

        return {
            "average_path_length": float(np.mean(path_lengths)),
            "average_path_smoothness": float(np.mean(path_smoothness)),
            "path_success_rate": float(num_paths / num_samples),
            "average_path_clearance": float(np.mean(path_clearances)),
        }

    def calculate_validation_metrics(self) -> dict[str, bool]:
        """
        Calculate metrics to validate graph correctness.

        Returns:
            Dictionary containing:
            - collision_free: Whether all edges are collision-free
            - node_validity: Whether all nodes are in valid positions
            - connectivity: Whether graph is fully connected
            - edge_length_validity: Whether edge lengths are reasonable
        """
        self._logger.info("Calculating validation metrics...")

        if len(self._graph) == 0:
            return {
                "collision_free": True,
                "node_validity": True,
                "connectivity": True,
                "edge_length_validity": True,
            }

        # Check node validity
        node_validity = all(self._is_valid_position(pixel) for _, pixel in self._graph.nodes(data="pixel"))

        # Check edge collisions and lengths
        max_edge_length = np.sqrt(self._occupancy_map.shape[0] ** 2 + self._occupancy_map.shape[1] ** 2)  # pixels
        edge_length_validity = True

        # Check all edge collisions at once
        collision_free = self._check_all_edges_collision()

        for _, _, data in self._graph.edges(data=True):
            # Check edge length
            if data["weight"] <= 0 or data["weight"] > max_edge_length:
                edge_length_validity = False
                break

        return {
            "collision_free": collision_free,
            "node_validity": node_validity,
            "connectivity": nx.is_connected(self._graph),
            "num_subgraphs": nx.number_connected_components(self._graph),
            "edge_length_validity": edge_length_validity,
        }

    def _check_all_edges_collision(self) -> bool:
        """Check if any edge collides with obstacles using image processing."""
        # Create a black image of the same size as the occupancy map
        edge_image = self._inflated_map

        # Draw all edges on the inflated map image as 0
        for n1, n2 in self._graph.edges():
            pixel1 = self._graph.nodes[n1]["pixel"]
            pixel2 = self._graph.nodes[n2]["pixel"]
            cv2.line(edge_image, pixel1[::-1], pixel2[::-1], color=0, thickness=1)  # Note: OpenCV uses (x, y) format
        # Check if any new white pixels appear in the combined image
        return np.all(edge_image == self._inflated_map).item()

    def _calculate_path_smoothness(self, path: list[tuple[int, int]]) -> float:
        """Calculate path smoothness using angles between consecutive edges."""
        if len(path) < 3:
            return 1.0  # Perfect smoothness for straight lines

        angles = []
        for i in range(len(path) - 2):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            p3 = np.array(path[i + 2])

            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            angles.append(angle)

        # Convert angles to smoothness metric (1 = smooth, 0 = sharp turns)
        smoothness = 1.0 - np.mean(angles) / np.pi
        return float(smoothness)

    def _calculate_path_clearance(self, path: list[tuple[int, int]]) -> float:
        """Calculate average clearance (distance to obstacles) along path using precomputed distance map."""
        if len(path) < 2:
            return 0.0

        def bresenham_line(x0, y0, x1, y1):
            """Generate points on a line using Bresenham's algorithm."""
            points = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                points.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return points

        clearances = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            # Get all points on the line segment between start and end
            line_points = bresenham_line(start[0], start[1], end[0], end[1])
            for y, x in line_points:
                if 0 <= y < self._distance_map.shape[0] and 0 <= x < self._distance_map.shape[1]:
                    clearance = self._distance_map[y, x]
                    clearances.append(clearance)

        return float(np.mean(clearances))

    def _is_valid_position(self, point: tuple[int, int]) -> bool:
        """Check if point is in bounds and in free space."""
        y, x = point
        return (
            0 <= y < self._occupancy_map.shape[0]
            and 0 <= x < self._occupancy_map.shape[1]
            and not self._inflated_map[y, x]
        )

    def _print_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        """Print all metrics in a readable format."""
        # Print all metrics in one statement
        print(
            f"""
=== Waypoint Graph Evaluation Summary ===
Coverage Metrics:
  - Free Space Coverage: {metrics['coverage']['free_space_coverage']:.2%}
  - Average Distance to Node: {metrics["coverage"]["average_distance_to_node"]:.2f} pixels

Graph Structure:
  - Number of Nodes: {metrics['graph_structure']['num_nodes']}
  - Number of Edges: {metrics['graph_structure']['num_edges']}
  - Average Node Degree: {metrics['graph_structure']['average_degree']:.2f}

Path Planning:
  - Path Success Rate: {metrics['path_planning']['path_success_rate']:.2%}
  - Average Path Smoothness: {metrics['path_planning']['average_path_smoothness']:.2f}
  - Average Path Clearance: {metrics['path_planning']['average_path_clearance']:.2f}

Validation:
  - Collision Free: {metrics['validation']['collision_free']}
  - Graph Connected: {metrics['validation']['connectivity']}
  - Number of Subgraphs: {metrics['validation']['num_subgraphs']}
=====================================
"""
        )
