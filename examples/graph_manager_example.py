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

"""Example showing how to use GraphManager to manage multiple maps."""

import os

import cv2
import numpy as np

from swagger import GraphManager, WaypointGraphGeneratorConfig
from swagger.models import Point


def create_warehouse_map(width=500, height=500, wall_thickness=10):
    """Create a synthetic warehouse map with open spaces and obstacles.

    Args:
        width: Width of the map in pixels
        height: Height of the map in pixels
        wall_thickness: Thickness of walls in pixels

    Returns:
        A numpy array representing the occupancy grid map
    """
    # Create empty map (all free space)
    map_image = np.ones((height, width), dtype=np.uint8) * 255

    # Add outer walls
    map_image[0:wall_thickness, :] = 0  # Top wall
    map_image[-wall_thickness:, :] = 0  # Bottom wall
    map_image[:, 0:wall_thickness] = 0  # Left wall
    map_image[:, -wall_thickness:] = 0  # Right wall

    # Add some internal walls and pillars
    pillar_size = 20
    # Horizontal wall
    map_image[200:220, 100:400] = 0
    # Vertical wall
    map_image[300:450, 350:370] = 0

    # Pillars
    for x, y in [(100, 100), (100, 400), (400, 100), (400, 400)]:
        # Split this into multiple lines to avoid line length limit
        y_min, y_max = y - pillar_size // 2, y + pillar_size // 2
        x_min, x_max = x - pillar_size // 2, x + pillar_size // 2
        map_image[y_min:y_max, x_min:x_max] = 0

    return map_image


def create_office_map(width=400, height=400, wall_thickness=10):
    """Create a synthetic office map with rooms and corridors.

    Args:
        width: Width of the map in pixels
        height: Height of the map in pixels
        wall_thickness: Thickness of walls in pixels

    Returns:
        A numpy array representing the occupancy grid map
    """
    # Create empty map (all free space)
    map_image = np.ones((height, width), dtype=np.uint8) * 255

    # Add outer walls
    map_image[0:wall_thickness, :] = 0
    map_image[-wall_thickness:, :] = 0
    map_image[:, 0:wall_thickness] = 0
    map_image[:, -wall_thickness:] = 0

    # Internal walls forming rooms and corridors
    # Horizontal walls
    map_image[100:110, 0:300] = 0
    map_image[200:210, 100:400] = 0
    map_image[300:310, 0:300] = 0

    # Vertical walls
    map_image[0:110, 200:210] = 0
    map_image[200:400, 100:110] = 0
    map_image[100:310, 300:310] = 0

    return map_image


def get_warehouse_config():
    """Create a configuration suitable for warehouse environments."""
    return WaypointGraphGeneratorConfig(
        skeleton_sample_distance=1.0,  # Distance between waypoints
        free_space_sampling_threshold=2.0,  # Sample free space from obstacles
        boundary_sample_distance=1.5,  # Sample boundaries
    )


def get_office_config():
    """Create a configuration suitable for office environments."""
    return WaypointGraphGeneratorConfig(
        skeleton_sample_distance=0.5,  # Closer waypoints for navigation
        free_space_sampling_threshold=1.0,  # Less free space in an office
        boundary_sample_distance=0.5,  # More samples along walls
        merge_node_distance=0.15,  # Merge nodes that are very close
    )


def save_maps(vis_dir, maps_info):
    """Save maps as images for reference.

    Args:
        vis_dir: Directory to save visualizations
        maps_info: List of (map_id, map_image) tuples
    """
    for map_id, map_image in maps_info:
        cv2.imwrite(os.path.join(vis_dir, f"{map_id}_map.png"), map_image)
    print("Saved map images to visualization directory")


def main():
    """Run the GraphManager example."""
    # Create a directory for visualizations
    vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Visualization files will be saved to: {vis_dir}")

    # Create a graph manager instance with the visualization directory
    graph_manager = GraphManager(visualization_dir=vis_dir)

    # Create sample maps
    map1_id = "warehouse"
    map1_image = create_warehouse_map()
    map1_resolution = 0.05  # meters/pixel
    map1_safety = 0.3  # meters

    map2_id = "office"
    map2_image = create_office_map()
    map2_resolution = 0.02  # meters/pixel (higher resolution)
    map2_safety = 0.25  # meters

    # Save the maps as images for reference
    save_maps(vis_dir, [(map1_id, map1_image), (map2_id, map2_image)])

    # Create the graphs with configurations for each environment
    print(f"Creating graph for {map1_id} map...")
    warehouse_graph = graph_manager.create_graph(
        map_id=map1_id,
        image=map1_image,
        resolution=map1_resolution,
        safety_distance=map1_safety,
        occupancy_threshold=127,
        config=get_warehouse_config(),
    )

    print(f"Creating graph for {map2_id} map...")
    graph_manager.create_graph(
        map_id=map2_id,
        image=map2_image,
        resolution=map2_resolution,
        safety_distance=map2_safety,
        occupancy_threshold=127,
        config=get_office_config(),
    )

    # Generate visualizations
    print("Generating graph visualizations...")
    graph_manager.visualize_graph(map_id=map1_id)
    graph_manager.visualize_graph(map_id=map2_id)
    print(f"Visualizations created at: {vis_dir}")

    # Find a route in the warehouse map
    print("Finding route in warehouse map...")
    warehouse_start = Point(x=4.7, y=15.0)
    warehouse_goal = Point(x=22.3, y=12.5)
    warehouse_route = graph_manager.find_route(map_id=map1_id, start=warehouse_start, goal=warehouse_goal)

    if warehouse_route:
        print(f"Found route with {len(warehouse_route)} waypoints")
        print(f"First few points: {warehouse_route[:3]}")
        print(f"Last few points: {warehouse_route[-3:]}")
    else:
        print("No route found in warehouse map")

    # Find nearest nodes in the office map
    print("Finding nearest nodes in office map...")
    office_points = [Point(x=3.0, y=4.0), Point(x=7.0, y=8.0)]
    office_node_ids = graph_manager.get_nearest_nodes(map_id=map2_id, points=office_points)
    print(f"Nearest node IDs: {office_node_ids}")

    # Get CSR graph for the warehouse map
    print("Getting graph for warehouse map...")
    warehouse_graph = graph_manager.get_graph(map_id=map1_id)
    if warehouse_graph:
        print(f"Graph has {warehouse_graph.number_of_nodes()} nodes and {warehouse_graph.number_of_edges()} edges")

    # Remove one map
    print(f"Removing {map2_id} map...")
    removed = graph_manager.remove_graph(map_id=map2_id)
    print(f"Map removed: {removed}")

    # Clear all maps
    print("Clearing all maps...")
    graph_manager.clear()
    print("All maps cleared")

    print("GraphManager example completed successfully!")


if __name__ == "__main__":
    main()
