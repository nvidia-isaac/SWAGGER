#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
Graph Builder with Coverage Path Planning

This script demonstrates incremental graph building with a planned exploration path
that covers the entire warehouse map. The robot follows a lawnmower pattern to
systematically explore all navigable areas.
"""

import os
import sys
import math
import cv2
import numpy as np
import json
import logging
import warnings
import time
from PIL import Image

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# Suppress logging from SWAGGER library
logging.getLogger('swagger.waypoint_graph_generator').setLevel(logging.ERROR)

# Set matplotlib backend before importing pyplot
import matplotlib
if os.environ.get('DISPLAY') is None or os.environ.get('DISPLAY') == '':
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

from swagger import (
    WaypointGraphGenerator,
    WaypointGraphGeneratorConfig,
    Point,
)
from visualize_global_graph_on_image import visualize_global_graph_on_image
import networkx as nx
from swagger.global_graph_generator import GlobalGraphGenerator
from networkx.readwrite import json_graph


# ============================================================
# Configuration
# ============================================================

RESOLUTION = 0.039          # meters per pixel
CROP_SIZE = 300             # Size of local frame (pixels)
SAFE_DIST = 0.30            # Robot safety distance (meters)

# Visualization settings
ENABLE_LIVE_VISUALIZATION = True
SAVE_GIF = True             # Save animation as GIF
GIF_FILENAME = "results/graph_building_animation.gif"
GIF_DURATION = 500          # Duration per frame in milliseconds

# Path planning parameters
OVERLAP_PIXELS = 120        # 40% overlap for good connectivity
VERTICAL_STEP = 150         # Balanced vertical steps to cover entire map


# ============================================================
# Coverage Path Planner
# ============================================================

def plan_coverage_path(map_image, crop_size, overlap, vertical_step):
    """
    Plan a systematic grid coverage path that explores the entire map.
    
    Returns list of (x_px, y_px, rotation_rad) tuples representing robot poses.
    """
    H, W = map_image.shape[:2]
    path = []
    
    # Calculate horizontal step (with overlap)
    horizontal_step = crop_size - overlap
    
    y = 0
    
    # Simple left-to-right, top-to-bottom grid pattern
    while y + crop_size <= H:
        x = 0
        while x + crop_size <= W:
            path.append((x, y, 0.0))  # All frames facing same direction
            x += horizontal_step
        
        # Move down to next row
        y += vertical_step
    
    return path


# ============================================================
# Frame Generator with Path
# ============================================================

def generate_frames_from_path(input_image_path: str, output_dir: str, 
                              path_waypoints: list, crop_size: int):
    """
    Generator that yields frames following a planned coverage path.
    
    Yields: (frame_idx, image_crop, filename, x_px, y_px, rotation_rad)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base map
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {input_image_path}")
    
    H, W = image.shape[:2]
    
    for i, (x_px, y_px, rotation_rad) in enumerate(path_waypoints):
        # Extract crop
        cropped = image[y_px:y_px + crop_size, x_px:x_px + crop_size]
        
        # Save frame
        crop_filename = os.path.join(output_dir, f"frame_{i:03d}.png")
        cv2.imwrite(crop_filename, cropped)
        
        yield i, cropped, crop_filename, x_px, y_px, rotation_rad


# ============================================================
# Visualization
# ============================================================

def visualize_graph_incremental(current_graph: nx.Graph, previous_graph: nx.Graph | None, 
                                 frame_idx: int, ax, base_image=None, resolution=0.039,
                                 save_path=None, robot_pose_px=None, all_robot_poses=None,
                                 previous_robot_pose=None):
    """
    Visualize the graph with robot position marker.
    """
    ax.clear()
    
    # Show base image
    if base_image is not None:
        ax.imshow(base_image, cmap='gray', origin='upper')
        ax.set_aspect('equal')
    
    if current_graph is None or current_graph.number_of_nodes() == 0:
        ax.set_title(f"Frame {frame_idx}: No nodes yet")
        return
    
    # Determine known vs new nodes
    known_nodes = set()
    new_nodes = set()
    
    if previous_graph is not None and previous_graph.number_of_nodes() > 0:
        known_nodes = set(previous_graph.nodes())
    
    for node in current_graph.nodes():
        if node not in known_nodes:
            new_nodes.add(node)
    
    # Convert world coordinates to pixel coordinates
    # World origin (0,0) is at the center of the first frame, which was at pixel position (0,0)
    # So world (0,0) maps to pixel (CROP_SIZE/2, CROP_SIZE/2) in the full map
    pos = {}
    
    world_origin_x_px = CROP_SIZE / 2  # World (0,0) is at this pixel position in full map
    world_origin_y_px = CROP_SIZE / 2
    
    for node, data in current_graph.nodes(data=True):
        if "world" in data:
            world = data["world"]
            x_m, y_m = world[0], world[1]
            
            # Reverse the transform:
            # We pass dy_m = -(y_px * res) to generator
            # Generator does: y_world = -(row - center) * res + dy_m
            #              = -(row - center) * res - y_px * res
            # For frame center (row=center): y_world = -y_px * res
            # So: y_px = -y_world / res
            # Then add world_origin offset: pixel_y = world_origin_y - y_world/res
            
            x_px = (x_m / resolution) + world_origin_x_px
            y_px = world_origin_y_px - (y_m / resolution)
            
            pos[node] = (x_px, y_px)
    
    # Debug: Print sample positions (only if needed for debugging)
    # if len(pos) > 0 and frame_idx == 7:  # Final frame
    #     ... debug code commented out
    
    # Draw edges
    known_edges = [(u, v) for u, v in current_graph.edges() if u in known_nodes and v in known_nodes]
    new_edges = [(u, v) for u, v in current_graph.edges() if u not in known_nodes or v not in known_nodes]
    
    for u, v in known_edges:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   'm-', linewidth=1.5, alpha=0.6, zorder=1)
    
    for u, v in new_edges:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   'b-', linewidth=2.0, alpha=0.8, zorder=2)
    
    # Draw nodes
    if known_nodes and pos:
        known_x = [pos[n][0] for n in known_nodes if n in pos]
        known_y = [pos[n][1] for n in known_nodes if n in pos]
        if known_x:
            ax.scatter(known_x, known_y, c='magenta', s=30, alpha=0.8, 
                      edgecolors='black', linewidths=0.5, zorder=3, label='Known')
    
    if new_nodes and pos:
        new_x = [pos[n][0] for n in new_nodes if n in pos]
        new_y = [pos[n][1] for n in new_nodes if n in pos]
        if new_x:
            ax.scatter(new_x, new_y, c='red', s=50, alpha=0.9, 
                      edgecolors='black', linewidths=1.0, zorder=4, label='New')
    
    # Draw robot position
    if robot_pose_px is not None:
        x_px, y_px, rotation = robot_pose_px
        # Robot at center of its local frame
        robot_x = x_px + CROP_SIZE / 2
        robot_y = y_px + CROP_SIZE / 2
        
        # Calculate actual movement direction if we have previous pose
        if previous_robot_pose is not None:
            prev_x, prev_y, _ = previous_robot_pose
            prev_center_x = prev_x + CROP_SIZE / 2
            prev_center_y = prev_y + CROP_SIZE / 2
            
            # Calculate angle from previous to current position
            dx = robot_x - prev_center_x
            dy = robot_y - prev_center_y
            if abs(dx) > 1 or abs(dy) > 1:  # Only update if moved significantly
                rotation = math.atan2(-dy, dx)  # Negative dy for image coordinates
        
        # Draw robot as triangle
        size = 20
        dx = size * math.cos(rotation)
        dy = -size * math.sin(rotation)  # Negative for image coordinates
        
        triangle = plt.Polygon([
            [robot_x + dx, robot_y + dy],
            [robot_x - dx/2 - dy/2, robot_y - dy/2 + dx/2],
            [robot_x - dx/2 + dy/2, robot_y - dy/2 - dx/2]
        ], color='green', alpha=0.8, zorder=10, label='Robot')
        ax.add_patch(triangle)
        
        # Draw field of view rectangle
        fov_rect = plt.Rectangle((x_px, y_px), CROP_SIZE, CROP_SIZE,
                                 linewidth=2, edgecolor='green', facecolor='none',
                                 alpha=0.5, zorder=9)
        ax.add_patch(fov_rect)
    
    ax.set_title(f"Frame {frame_idx}: {len(current_graph.nodes())} nodes "
                f"({len(new_nodes)} new, {len(known_nodes)} known)")
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('equal')
    
    if base_image is not None:
        ax.set_xlim(0, base_image.shape[1])
        ax.set_ylim(base_image.shape[0], 0)  # Inverted Y-axis like graph-builder.py
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    # Always update display if interactive
    if not matplotlib.get_backend() == 'Agg':
        plt.draw()
        plt.pause(0.5)  # Increased pause to observe frame transitions


# ============================================================
# Main
# ============================================================

def main():
    # Clean up old results
    output_dir = "results"
    maps_dir = "results/simulated_route"
    
    if os.path.exists(maps_dir):
        import shutil
        shutil.rmtree(maps_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base map for path planning
    base_map_path = "../maps/carter_warehouse_navigation.png"
    base_map = cv2.imread(base_map_path, cv2.IMREAD_GRAYSCALE)
    if base_map is None:
        raise FileNotFoundError(f"Could not load: {base_map_path}")
    
    # Plan coverage path
    coverage_path = plan_coverage_path(
        base_map, 
        crop_size=CROP_SIZE,
        overlap=OVERLAP_PIXELS,
        vertical_step=VERTICAL_STEP
    )
    
    # Configure SWAGGER - Adjusted for less dense graph
    config = WaypointGraphGeneratorConfig(
        skeleton_sample_distance=0.25,      # Increased from 0.20 (fewer skeleton nodes)
        boundary_inflation_factor=2,
        boundary_sample_distance=0.40,      # Increased from 0.30 (fewer boundary nodes)
        free_space_sampling_threshold=0.65, # Increased from 0.50 (more selective sampling)
        merge_node_distance=0.50,           # Increased to better merge nearby nodes (was 0.30)
        min_subgraph_length=0.50,           # Increased from 0.35 (prune shorter paths)
        use_skeleton_graph=False,
        use_boundary_sampling=True,
        use_free_space_sampling=True,
        use_delaunay_shortcuts=True,        # Re-enabled for cross-connections between regions
        prune_graph=True,
        debug=False,
    )
    
    # Create generator with WARNING level logging to suppress INFO messages
    generator = WaypointGraphGenerator(config, logger_level=logging.WARNING)
    global_builder = GlobalGraphGenerator(
        merge_distance=0.50,  # Increased to better merge overlapping frames (0.5m = ~12.8 pixels)
        retention_factor=1.0
    )
    
    persisted_known_points = []
    
    # Setup visualization
    fig, ax = None, None
    is_headless = matplotlib.get_backend() == 'Agg'
    gif_frames = []  # Store frames for GIF creation
    
    if ENABLE_LIVE_VISUALIZATION:
        if is_headless or SAVE_GIF:
            os.makedirs("results/viz_frames", exist_ok=True)
        if not is_headless:
            plt.ion()
        
        # Set figure size to match map aspect ratio (480W x 776H = 0.618 aspect)
        map_aspect = 480 / 776  # width / height
        fig_height = 16
        fig_width = fig_height * map_aspect
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    previous_global_graph = None
    previous_robot_pose = None  # Track previous robot position for direction
    
    # Generate frames and build graph
    frame_generator = generate_frames_from_path(
        base_map_path,
        maps_dir,
        coverage_path,
        CROP_SIZE
    )
    
    for i, occupancy, img_path, x_px, y_px, rotation_rad in frame_generator:
        # Only print frame number and position
        print(f"Frame {i}/{len(coverage_path)-1}: position=({x_px}, {y_px})px")
        
        # Compute world offset
        # CRITICAL: Generator uses ROS convention where Y+ points UP
        # But image pixel Y+ points DOWN
        # So we need to NEGATE y_px when converting to world offset
        dx_m = x_px * RESOLUTION
        dy_m = -(y_px * RESOLUTION)  # NEGATE for ROS convention
        
        # Build local graph
        robot_pose = (dx_m, dy_m, 0.0)  # Force rotation=0 for world coordinate alignment
        
        # Build local graph (library debug output suppressed via config.debug=False)
        local_start = time.time()
        local_graph = generator.build_graph_from_grid_map(
            image=occupancy,
            resolution=RESOLUTION,
            x_offset=robot_pose[0],
            y_offset=robot_pose[1],
            rotation=robot_pose[2],
            known_points=persisted_known_points,
            safety_distance=SAFE_DIST,
            occupancy_threshold=127,
        )
        local_time = time.time() - local_start
        print(f"  └─ Local graph: {local_time*1000:.1f}ms ({local_graph.number_of_nodes()} nodes, {local_graph.number_of_edges()} edges)")
        
        if local_graph is None:
            continue
        
        # Debug: Check local graph node coordinates (commented out)
        # if i == 5 and local_graph.number_of_nodes() > 0:
        #     ... debug code
        
        # Merge into global graph
        global_start = time.time()
        global_builder.add_local_graph(local_graph)
        current_global = global_builder.get_global_graph()
        global_time = time.time() - global_start
        print(f"  └─ Global merge: {global_time*1000:.1f}ms (total: {current_global.number_of_nodes()} nodes, {current_global.number_of_edges()} edges)\n")
        
        # Debug: Check global graph node coordinates (commented out)
        # if i == 5 and current_global.number_of_nodes() > 0:
        #     ... debug code
        
        # Visualize
        if ENABLE_LIVE_VISUALIZATION and ax is not None:
            save_path = f"results/viz_frames/frame_{i:03d}.png" if (is_headless or SAVE_GIF) else None
            
            visualize_graph_incremental(
                current_graph=current_global,
                previous_graph=previous_global_graph,
                frame_idx=i,
                ax=ax,
                base_image=base_map,
                resolution=RESOLUTION,
                save_path=save_path,
                robot_pose_px=(x_px, y_px, rotation_rad),
                previous_robot_pose=previous_robot_pose
            )
            
            # Update previous pose for next iteration
            previous_robot_pose = (x_px, y_px, rotation_rad)
            
            # If saving GIF, capture frame
            if SAVE_GIF:
                # Convert matplotlib figure to PIL Image
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif_frames.append(Image.fromarray(img_array))
            
            previous_global_graph = current_global.copy()
    
    # Save final results
    current_global = global_builder.get_global_graph()
    
    # Save visualization
    if isinstance(current_global, nx.Graph) and current_global.number_of_nodes() > 0:
        overlay_path = os.path.join(output_dir, "global_graph_overlay.png")
        visualize_global_graph_on_image(
            current_global,
            base_image_path=base_map_path,
            output_path=overlay_path,
        )
    
    # Save JSON
    def _make_json_serializable(obj):
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except:
            pass
        if isinstance(obj, dict):
            return {str(k): _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        try:
            return float(obj)
        except:
            return str(obj)
    
    mapping = {n: i for i, n in enumerate(current_global.nodes())}
    G_export = nx.relabel_nodes(current_global, mapping, copy=True)
    
    json_path = os.path.join(output_dir, "global_graph.json")
    node_link = json_graph.node_link_data(G_export, edges="links")
    with open(json_path, "w") as f:
        json.dump(_make_json_serializable(node_link), f, indent=2)
    
    # Print final summary
    print(f"\nComplete! {len(coverage_path)} frames → {len(current_global.nodes())} nodes, {len(current_global.edges())} edges")
    
    # Check for potential issues
    degrees = [d for n, d in current_global.degree()]
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    print(f"  └─ Max degree: {max_degree}, Avg degree: {avg_degree:.2f}")
    print(f"  └─ Self-loops: {len(list(nx.selfloop_edges(current_global)))}")
    
    # Save GIF animation
    if SAVE_GIF and gif_frames:
        print(f"\nSaving animation GIF with {len(gif_frames)} frames...")
        gif_frames[0].save(
            GIF_FILENAME,
            save_all=True,
            append_images=gif_frames[1:],
            duration=GIF_DURATION,
            loop=0
        )
        print(f"  └─ Saved to {GIF_FILENAME}")
    
    if ENABLE_LIVE_VISUALIZATION and fig is not None and not is_headless:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
