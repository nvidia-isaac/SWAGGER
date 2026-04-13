#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
Path Planning Visualization

This script loads a saved navigation graph and visualizes A* path planning
with an animated robot moving from start to goal positions.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
from networkx.readwrite import json_graph

# Configuration
GRAPH_PATH = "results/global_graph.json"
MAP_PATH = "../maps/carter_warehouse_navigation.png"
RESOLUTION = 0.039  # meters per pixel

def load_graph(graph_path):
    """Load the navigation graph from JSON."""
    with open(graph_path, 'r') as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data, edges="links")
    return G

def world_to_pixel(world_coords, resolution, crop_size=300):
    """Convert world coordinates to pixel coordinates for visualization."""
    x_m, y_m = world_coords[0], world_coords[1]
    world_origin_x_px = crop_size / 2
    world_origin_y_px = crop_size / 2
    
    x_px = (x_m / resolution) + world_origin_x_px
    y_px = world_origin_y_px - (y_m / resolution)
    
    return x_px, y_px

def find_nearest_node(graph, target_pixel):
    """Find the nearest graph node to a target pixel location."""
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in graph.nodes(data=True):
        world = data.get('world')
        if world is None:
            continue
        
        node_px = world_to_pixel(world, RESOLUTION)
        dist = np.hypot(node_px[0] - target_pixel[0], node_px[1] - target_pixel[1])
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

def compute_path(graph, start_node, goal_node):
    """Compute shortest path using A* algorithm."""
    def heuristic(n1, n2):
        w1 = graph.nodes[n1]['world']
        w2 = graph.nodes[n2]['world']
        return np.hypot(w1[0] - w2[0], w1[1] - w2[1])
    
    try:
        path = nx.astar_path(graph, start_node, goal_node, heuristic=heuristic, weight='weight')
        return path
    except nx.NetworkXNoPath:
        print("No path found!")
        return None

def interpolate_path(path_nodes, graph, num_points=50):
    """Interpolate smooth positions along the path for animation."""
    positions = []
    
    for i in range(len(path_nodes) - 1):
        n1 = path_nodes[i]
        n2 = path_nodes[i + 1]
        
        p1 = world_to_pixel(graph.nodes[n1]['world'], RESOLUTION)
        p2 = world_to_pixel(graph.nodes[n2]['world'], RESOLUTION)
        
        for t in np.linspace(0, 1, num_points, endpoint=False):
            x = p1[0] * (1 - t) + p2[0] * t
            y = p1[1] * (1 - t) + p2[1] * t
            positions.append((x, y))
    
    # Add final position
    final_pos = world_to_pixel(graph.nodes[path_nodes[-1]]['world'], RESOLUTION)
    positions.append(final_pos)
    
    return positions

def visualize_path_planning(graph, map_image, start_pixel, goal_pixel):
    """Create animated visualization of path planning."""
    
    # Find nearest nodes
    start_node = find_nearest_node(graph, start_pixel)
    goal_node = find_nearest_node(graph, goal_pixel)
    
    if start_node is None or goal_node is None:
        print("Could not find start or goal node")
        return
    
    print(f"Start node: {start_node}")
    print(f"Goal node: {goal_node}")
    
    # Compute path
    path = compute_path(graph, start_node, goal_node)
    
    if path is None:
        return
    
    print(f"Path length: {len(path)} nodes")
    
    # Interpolate positions for smooth animation
    positions = interpolate_path(path, graph, num_points=20)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(map_image, cmap='gray', origin='upper')
    ax.set_aspect('equal')
    ax.set_xlim(0, map_image.shape[1])
    ax.set_ylim(map_image.shape[0], 0)
    
    # Draw graph edges (light)
    for u, v in graph.edges():
        w1 = graph.nodes[u].get('world')
        w2 = graph.nodes[v].get('world')
        if w1 and w2:
            p1 = world_to_pixel(w1, RESOLUTION)
            p2 = world_to_pixel(w2, RESOLUTION)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Draw graph nodes (small)
    for node, data in graph.nodes(data=True):
        world = data.get('world')
        if world:
            px = world_to_pixel(world, RESOLUTION)
            ax.plot(px[0], px[1], 'o', color='lightblue', markersize=2, alpha=0.5, zorder=2)
    
    # Draw planned path (highlighted)
    path_pixels = [world_to_pixel(graph.nodes[n]['world'], RESOLUTION) for n in path]
    path_x = [p[0] for p in path_pixels]
    path_y = [p[1] for p in path_pixels]
    ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, zorder=3, label='Planned Path')
    
    # Draw waypoints on path
    for px in path_pixels:
        ax.plot(px[0], px[1], 'o', color='blue', markersize=6, zorder=4)
    
    # Draw start and goal
    ax.plot(start_pixel[0], start_pixel[1], 'go', markersize=15, label='Start', zorder=5)
    ax.plot(goal_pixel[0], goal_pixel[1], 'ro', markersize=15, label='Goal', zorder=5)
    
    # Robot marker (to be animated)
    robot, = ax.plot([], [], 'o', color='lime', markersize=20, zorder=10, 
                     markeredgecolor='darkgreen', markeredgewidth=2, label='Robot')
    trail, = ax.plot([], [], 'g-', linewidth=2, alpha=0.5, zorder=9)
    
    # Trail storage
    trail_x = []
    trail_y = []
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Path Planning Visualization')
    plt.tight_layout()
    
    def init():
        robot.set_data([], [])
        trail.set_data([], [])
        return robot, trail
    
    def animate(frame):
        if frame < len(positions):
            x, y = positions[frame]
            robot.set_data([x], [y])
            
            trail_x.append(x)
            trail_y.append(y)
            trail.set_data(trail_x, trail_y)
        
        return robot, trail
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions),
                        interval=50, blit=True, repeat=True)
    
    plt.show()

def main():
    # Load graph and map
    print("Loading graph...")
    graph = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    print("Loading map...")
    map_image = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    
    # Define start and goal positions (pixel coordinates)
    # You can modify these to test different paths
    start_pixel = (150, 200)  # Upper left area
    goal_pixel = (400, 600)   # Lower right area
    
    print(f"Planning path from {start_pixel} to {goal_pixel}")
    
    # Visualize
    visualize_path_planning(graph, map_image, start_pixel, goal_pixel)

if __name__ == "__main__":
    main()
