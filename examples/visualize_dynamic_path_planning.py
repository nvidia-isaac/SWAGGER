#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
Dynamic Path Planning Visualization

This script visualizes path planning with dynamic graph updates. As the robot
moves, it periodically reloads the graph and replans the path if the graph changes.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from matplotlib.animation import FuncAnimation
import time

# Configuration
GRAPH_PATH = "results/global_graph.json"
MAP_PATH = "../maps/carter_warehouse_navigation.png"
RESOLUTION = 0.039  # meters per pixel
REPLAN_INTERVAL = 30  # Replan every N frames (~1.5 seconds at 50ms interval)

class DynamicPathPlanner:
    def __init__(self, graph_path, map_path, start_pixel, goal_pixel):
        self.graph_path = graph_path
        self.map_path = map_path
        self.start_pixel = start_pixel
        self.goal_pixel = goal_pixel
        
        # Load initial data
        self.graph = self.load_graph()
        self.map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        
        # Find start and goal nodes
        self.start_node = self.find_nearest_node(start_pixel)
        self.goal_node = self.find_nearest_node(goal_pixel)
        
        # Current state
        self.current_path = None
        self.current_positions = []
        self.position_index = 0
        self.replan_counter = 0
        self.trail_x = []
        self.trail_y = []
        
        # Initial path planning
        self.replan()
        
        print(f"Initial path: {len(self.current_path)} nodes" if self.current_path else "No initial path found")
    
    def load_graph(self):
        """Load the navigation graph from JSON."""
        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data, edges="links")
            return G
        except Exception as e:
            print(f"Error loading graph: {e}")
            return nx.Graph()
    
    def world_to_pixel(self, world_coords, crop_size=300):
        """Convert world coordinates to pixel coordinates."""
        x_m, y_m = world_coords[0], world_coords[1]
        world_origin_x_px = crop_size / 2
        world_origin_y_px = crop_size / 2
        
        x_px = (x_m / RESOLUTION) + world_origin_x_px
        y_px = world_origin_y_px - (y_m / RESOLUTION)
        
        return x_px, y_px
    
    def find_nearest_node(self, target_pixel):
        """Find the nearest graph node to a target pixel location."""
        min_dist = float('inf')
        nearest_node = None
        
        for node, data in self.graph.nodes(data=True):
            world = data.get('world')
            if world is None:
                continue
            
            node_px = self.world_to_pixel(world)
            dist = np.hypot(node_px[0] - target_pixel[0], node_px[1] - target_pixel[1])
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node
    
    def compute_path(self, start_node, goal_node):
        """Compute shortest path using A* algorithm."""
        def heuristic(n1, n2):
            w1 = self.graph.nodes[n1]['world']
            w2 = self.graph.nodes[n2]['world']
            return np.hypot(w1[0] - w2[0], w1[1] - w2[1])
        
        try:
            path = nx.astar_path(self.graph, start_node, goal_node, heuristic=heuristic, weight='weight')
            return path
        except (nx.NetworkXNoPath, KeyError):
            return None
    
    def interpolate_path(self, path_nodes, num_points=20):
        """Interpolate smooth positions along the path."""
        positions = []
        
        for i in range(len(path_nodes) - 1):
            n1 = path_nodes[i]
            n2 = path_nodes[i + 1]
            
            p1 = self.world_to_pixel(self.graph.nodes[n1]['world'])
            p2 = self.world_to_pixel(self.graph.nodes[n2]['world'])
            
            for t in np.linspace(0, 1, num_points, endpoint=False):
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                positions.append((x, y))
        
        # Add final position
        final_pos = self.world_to_pixel(self.graph.nodes[path_nodes[-1]]['world'])
        positions.append(final_pos)
        
        return positions
    
    def replan(self):
        """Reload graph and replan path."""
        print("Replanning...")
        
        # Reload graph
        old_num_nodes = len(self.graph.nodes) if self.graph else 0
        self.graph = self.load_graph()
        new_num_nodes = len(self.graph.nodes)
        
        if new_num_nodes != old_num_nodes:
            print(f"Graph updated: {old_num_nodes} → {new_num_nodes} nodes")
        
        # Update start node to current position if we're already moving
        if self.current_positions and self.position_index < len(self.current_positions):
            current_pos = self.current_positions[self.position_index]
            self.start_node = self.find_nearest_node(current_pos)
        else:
            self.start_node = self.find_nearest_node(self.start_pixel)
        
        # Always update goal node in case it changed
        self.goal_node = self.find_nearest_node(self.goal_pixel)
        
        # Compute new path
        self.current_path = self.compute_path(self.start_node, self.goal_node)
        
        if self.current_path:
            self.current_positions = self.interpolate_path(self.current_path)
            self.position_index = 0
            print(f"New path: {len(self.current_path)} nodes")
        else:
            print("No path found during replan")
    
    def get_current_position(self):
        """Get the current robot position."""
        if not self.current_positions or self.position_index >= len(self.current_positions):
            return None
        return self.current_positions[self.position_index]
    
    def step(self):
        """Advance one step in the animation."""
        self.replan_counter += 1
        
        # Check if we need to replan
        if self.replan_counter >= REPLAN_INTERVAL:
            self.replan()
            self.replan_counter = 0
        
        # Advance position
        if self.current_positions and self.position_index < len(self.current_positions):
            self.position_index += 1
            
            # If reached end, loop back
            if self.position_index >= len(self.current_positions):
                self.position_index = 0
                self.trail_x.clear()
                self.trail_y.clear()

def visualize_dynamic_planning(planner):
    """Create animated visualization with dynamic replanning."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Robot marker and trail
    robot, = ax.plot([], [], 'o', color='lime', markersize=20, zorder=10, 
                     markeredgecolor='darkgreen', markeredgewidth=2, label='Robot')
    trail, = ax.plot([], [], 'g-', linewidth=2, alpha=0.5, zorder=9)
    path_line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.7, zorder=3, label='Current Path')
    
    # Static elements
    ax.imshow(planner.map_image, cmap='gray', origin='upper')
    ax.set_aspect('equal')
    ax.set_xlim(0, planner.map_image.shape[1])
    ax.set_ylim(planner.map_image.shape[0], 0)
    
    # Draw start and goal
    ax.plot(planner.start_pixel[0], planner.start_pixel[1], 'go', markersize=15, 
            label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(planner.goal_pixel[0], planner.goal_pixel[1], 'ro', markersize=15, 
            label='Goal', zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    ax.legend(loc='upper right', fontsize=10)
    title = ax.set_title('Dynamic Path Planning')
    plt.tight_layout()
    
    def init():
        robot.set_data([], [])
        trail.set_data([], [])
        path_line.set_data([], [])
        return robot, trail, path_line, title
    
    def animate(frame):
        # Step the planner
        planner.step()
        
        # Update graph visualization (edges - light)
        ax.clear()
        ax.imshow(planner.map_image, cmap='gray', origin='upper')
        ax.set_aspect('equal')
        ax.set_xlim(0, planner.map_image.shape[1])
        ax.set_ylim(planner.map_image.shape[0], 0)
        
        # Draw graph edges
        for u, v in planner.graph.edges():
            w1 = planner.graph.nodes[u].get('world')
            w2 = planner.graph.nodes[v].get('world')
            if w1 and w2:
                p1 = planner.world_to_pixel(w1)
                p2 = planner.world_to_pixel(w2)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Draw current path
        if planner.current_path:
            path_pixels = [planner.world_to_pixel(planner.graph.nodes[n]['world']) 
                          for n in planner.current_path]
            path_x = [p[0] for p in path_pixels]
            path_y = [p[1] for p in path_pixels]
            ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, zorder=3, label='Current Path')
            
            # Draw waypoints
            for px in path_pixels:
                ax.plot(px[0], px[1], 'o', color='blue', markersize=4, zorder=4)
        
        # Draw start and goal
        ax.plot(planner.start_pixel[0], planner.start_pixel[1], 'go', markersize=15, 
                label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(planner.goal_pixel[0], planner.goal_pixel[1], 'ro', markersize=15, 
                label='Goal', zorder=5, markeredgecolor='darkred', markeredgewidth=2)
        
        # Update robot position
        pos = planner.get_current_position()
        if pos:
            x, y = pos
            ax.plot([x], [y], 'o', color='lime', markersize=20, zorder=10, 
                   markeredgecolor='darkgreen', markeredgewidth=2, label='Robot')
            
            planner.trail_x.append(x)
            planner.trail_y.append(y)
            ax.plot(planner.trail_x, planner.trail_y, 'g-', linewidth=2, alpha=0.5, zorder=9)
        
        # Update title with stats
        num_nodes = len(planner.graph.nodes) if planner.graph else 0
        path_len = len(planner.current_path) if planner.current_path else 0
        ax.set_title(f'Dynamic Path Planning (Graph: {num_nodes} nodes, Path: {path_len} waypoints)')
        
        ax.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=None,
                        interval=50, blit=False, repeat=True)
    
    plt.show()

def main():
    # Define start and goal positions (pixel coordinates)
    start_pixel = (150, 200)  # Upper left area
    goal_pixel = (400, 600)   # Lower right area
    
    print("Initializing dynamic path planner...")
    planner = DynamicPathPlanner(GRAPH_PATH, MAP_PATH, start_pixel, goal_pixel)
    
    print(f"Starting visualization (replanning every {REPLAN_INTERVAL} frames)...")
    visualize_dynamic_planning(planner)

if __name__ == "__main__":
    main()
