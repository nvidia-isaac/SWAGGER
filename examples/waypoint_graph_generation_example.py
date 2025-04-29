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
from pathlib import Path

import cv2

from swagger import Point, WaypointGraphGenerator, WaypointGraphGeneratorConfig


def main():
    # Prepare the required data
    occupancy_grid = cv2.imread(
        Path(__file__).parent.parent / "data" / "carter_warehouse_navigation.png", cv2.IMREAD_GRAYSCALE
    )
    occupancy_threshold = 127
    safety_distance = 0.3
    resolution = 0.05
    x_offset = 0.0
    y_offset = 0.0
    rotation = 0.0

    # Create a custom configuration
    config = WaypointGraphGeneratorConfig(
        skeleton_sample_distance=1.0,  # Distance between waypoints (in meters)
        boundary_inflation_factor=2.0,  # Larger margins around obstacles
        boundary_sample_distance=1.5,  # Distance between boundary samples (in meters)
        free_space_sampling_threshold=2.0,  # Distance from obstacles for free space sampling (in meters)
        merge_node_distance=0.3,  # Distance to merge nearby nodes (in meters)
        min_subgraph_length=1.0,  # Minimum subgraph length to keep (in meters)
        use_skeleton_graph=True,  # Create a graph along the medial axis
        use_boundary_sampling=True,  # Sample nodes along boundaries
        use_free_space_sampling=True,  # Sample nodes in open areas
        use_delaunay_shortcuts=True,  # Create shortcut edges
        prune_graph=True,  # Remove redundant nodes and edges
    )

    # Create generator with custom config
    generator = WaypointGraphGenerator(config=config)

    # Generate graph with the custom configuration
    graph = generator.build_graph_from_grid_map(
        image=occupancy_grid,
        occupancy_threshold=occupancy_threshold,
        safety_distance=safety_distance,
        resolution=resolution,
        x_offset=x_offset,
        y_offset=y_offset,
        rotation=rotation,
    )

    print(f"Node 1: {graph.nodes[1]}")
    print(f"Edge (0, 1): {graph.edges[(0, 1)]}")

    # Visualize the graph
    output_dir = "results"
    output_filename = "waypoint_graph_example.png"
    os.makedirs(output_dir, exist_ok=True)
    generator.visualize_graph(output_dir=output_dir, output_filename=output_filename)

    # Find a route between two points
    start = Point(x=13.0, y=18.0)
    goal = Point(x=15.0, y=10.0)
    route = generator.find_route(start, goal)
    print(f"Route: {[(point.x, point.y) for point in route]}")

    # Get node IDs for a list of points
    points = [Point(x=13.0, y=18.0), Point(x=15.0, y=10.0), Point(x=0.0, y=0.0)]
    node_ids = generator.get_node_ids(points)
    print(f"Node IDs: {node_ids}")


if __name__ == "__main__":
    main()
