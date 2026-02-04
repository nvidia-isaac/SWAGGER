#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
Graph Builder Example - Incremental waypoint graph generation with rotation support

This script demonstrates incremental graph building from simulated robot exploration.
For Isaac Lab integration, replace the frame generator with:

    # In Isaac Lab simulation loop:
    robot_x, robot_y, robot_yaw = get_robot_pose()  # Get from Isaac Lab
    elevation_map = get_elevation_map_crop(robot_x, robot_y, crop_size=256)  # Get from Isaac Lab
    
    # Convert to occupancy grid (threshold elevation, inflate obstacles, etc.)
    occupancy_grid = elevation_to_occupancy(elevation_map)
    
    # Pass to graph generator with actual robot pose
    robot_pose_meters = (robot_x, robot_y, robot_yaw)
    local_graph = generator.build_graph_from_grid_map(
        image=occupancy_grid,
        resolution=0.039,  # meters per pixel
        x_offset=robot_x,
        y_offset=robot_y,
        rotation=robot_yaw,  # IMPORTANT: Include rotation!
        known_points=known_points_from_previous_frames
    )
    
    # Merge into global graph
    global_builder.add_local_graph(local_graph)
"""

import os
import glob
import math
from pathlib import Path

import cv2
import numpy as np
import json
import os
import sys

# Set matplotlib backend before importing pyplot
import matplotlib
# Check if running in headless environment
if os.environ.get('DISPLAY') is None or os.environ.get('DISPLAY') == '':
    matplotlib.use('Agg')  # Non-interactive backend for headless systems
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
from swagger.graph_manager import GraphManager
from simulate_robot_graph_route import simulate_robot_graph_route
from visualize_global_graph_on_image import visualize_global_graph_on_image
import networkx as nx
from swagger.global_graph_generator import GlobalGraphGenerator
from networkx.readwrite import json_graph


# ============================================================
# Configuration
# ============================================================

RESOLUTION = 0.039          # meters per pixel
STEP_PIXELS = 100           # how far robot moves per frame (px)
MAX_EDGE_DIST = 2.0         # not always used
SAFE_DIST = 0.30            # used for sampling/collision distance
USE_LEGACY_GENERATOR = True  # toggle between current and legacy sampling logic

# Visualization settings
ENABLE_LIVE_VISUALIZATION = True  # Set to False to disable real-time plotting

# Export toggles for JSON output (controls what gets streamed)
# If you set these to True, the exporter will include pixel/pos/origin in the
# id_map; otherwise they are omitted to save bandwidth.
EXPORT_PIXEL = False
EXPORT_POS = False
EXPORT_ORIGIN = False


# ============================================================
# Utility Functions
# ============================================================

def generate_robot_frames(input_image_path: str, output_dir: str, num_frames: int,
                         crop_width: int, crop_height: int, step_size: int, 
                         motion_direction: str = "x", rotation: float = 0.0):
    """
    Generator that yields frames one at a time instead of pre-generating all.
    Useful for simulating real-time exploration.
    
    Calculates valid frames based on map size and yields: (frame_idx, image, filename, x_offset_px, y_offset_px, rotation)
    
    Args:
        rotation: Rotation in radians (default 0.0 for pure translation)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base map
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {input_image_path}")
    H, W = image.shape[:2]
    
    # Calculate how many valid frames we can actually generate
    if motion_direction == "x":
        max_frames = (W - crop_width) // step_size + 1
    elif motion_direction == "y":
        max_frames = (H - crop_height) // step_size + 1
    elif motion_direction == "xy":
        max_frames = min((W - crop_width) // step_size, (H - crop_height) // step_size) + 1
    else:
        raise ValueError("motion_direction must be 'x', 'y', or 'xy'")
    
    actual_frames = min(num_frames, max_frames)
    print(f"[INFO] Map size: {W}x{H}, crop: {crop_width}x{crop_height}, step: {step_size}")
    print(f"[INFO] Requested {num_frames} frames, generating {actual_frames} valid frames")
    
    for i in range(actual_frames):
        # Calculate crop region
        if motion_direction == "x":
            x_start, y_start = i * step_size, 0
        elif motion_direction == "y":
            x_start, y_start = 0, i * step_size
        elif motion_direction == "xy":
            x_start, y_start = i * step_size, i * step_size
        else:
            raise ValueError("motion_direction must be 'x', 'y', or 'xy'")
        
        cropped = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
        crop_filename = os.path.join(output_dir, f"frame_{i:03d}.png")
        cv2.imwrite(crop_filename, cropped)
        
        print(f"[INFO] Generated frame {i}/{actual_frames-1} (x_offset={x_start}px, y_offset={y_start}px)")
        
        # Yield frame index, image, filename, AND pixel offsets, AND rotation
        yield i, cropped, crop_filename, x_start, y_start, rotation


def offset_graph(local_graph: nx.Graph, dx_m: float, dy_m: float):
    """
    Apply world-coordinate offset to every node in the local graph.
    """
    for n, data in local_graph.nodes(data=True):
        if "world" in data:
            x, y, z = data["world"]
            data["world"] = (x + dx_m, y + dy_m, z)
    return local_graph


def offset_known_points(points, dx_m, dy_m):
    """
    Returns a NEW list of Point() with world-coordinate offsets applied.
    """
    out = []
    for p in points:
        out.append(Point(x=p.x + dx_m, y=p.y + dy_m, z=p.z))
    return out


def ensure_graph_pos_consistent(graph: nx.Graph):
    """
    Forces every graph node to have a consistent 'pos' = (x, y) in meters.
    """
    for node, data in graph.nodes(data=True):
        if "world" in data:
            world = data["world"]
            if len(world) == 3:
                x, y, _ = world
            elif len(world) == 2:
                x, y = world
            else:
                # Skip malformed entries but leave debug breadcrumbs.
                print(f"[WARN] Node {node} has unexpected world data: {world}")
                continue
            data["pos"] = (float(x), float(y))
        elif "pixel" in data:
            px, py = data["pixel"]
            data["pos"] = (px * RESOLUTION, py * RESOLUTION)
    return graph


def smart_merge_graphs(global_graph: nx.Graph | None, local_graph: nx.Graph):
    """
    Merge local_graph into global_graph WITHOUT duplicating nodes or
    creating self-loops. This is the same logic you were using before.
    """
    if global_graph is None or len(global_graph) == 0:
        return local_graph.copy()

    # Add nodes
    for n, d in local_graph.nodes(data=True):
        if n not in global_graph:
            global_graph.add_node(n, **d)

    # Add edges
    for u, v, ed in local_graph.edges(data=True):
        if not global_graph.has_edge(u, v):
            global_graph.add_edge(u, v, **ed)

    # Clean up bad edges
    global_graph.remove_edges_from(nx.selfloop_edges(global_graph))

    return global_graph


def visualize_graph_incremental(current_graph: nx.Graph, previous_graph: nx.Graph | None, 
                                 frame_idx: int, ax, base_image=None, resolution=0.039,
                                 save_path=None):
    """
    Visualize the graph with color coding:
    - Magenta: Known nodes from previous iterations
    - Red: Newly added nodes
    - Magenta: Edges between known nodes
    - Blue: New edges (at least one endpoint is new)
    
    If save_path is provided, saves the figure instead of showing it interactively.
    """
    import matplotlib.pyplot as plt
    
    ax.clear()
    
    # Show base image if provided
    if base_image is not None:
        ax.imshow(base_image, cmap='gray', origin='upper')
        ax.set_aspect('equal')
    
    if current_graph is None or current_graph.number_of_nodes() == 0:
        ax.set_title(f"Frame {frame_idx}: No nodes yet")
        return
    
    # Determine which nodes are known vs new
    known_nodes = set()
    new_nodes = set()
    
    if previous_graph is not None and previous_graph.number_of_nodes() > 0:
        known_nodes = set(previous_graph.nodes())
    
    for node in current_graph.nodes():
        if node not in known_nodes:
            new_nodes.add(node)
    
    # Prepare positions for plotting
    # The generator uses a ROS-style centered coordinate system where world (0,0) is at the center of each frame
    # We need to convert world coordinates (meters) to pixel coordinates accounting for this
    pos = {}
    
    # Each frame is 256x256, so center is at (128, 128) in local frame coordinates
    # But we're displaying on the full base map, so we need to account for centering in each frame
    frame_height = 256
    frame_width = 256
    
    for node, data in current_graph.nodes(data=True):
        if "world" in data:
            world = data["world"]
            x_m, y_m = world[0], world[1]
            
            # Convert world coordinates (meters) to pixels
            # The generator's coordinate system has (0,0) at the center of the frame
            # x is horizontal (left-to-right), y is vertical (but flipped: positive is up, negative is down)
            x_px = x_m / resolution
            y_px = -y_m / resolution  # Flip y-axis: positive y in world = up = negative in image coords
            
            # Add center offset: world (0,0) is at pixel (128, 128) in the local frame
            x_px += frame_width / 2
            y_px += frame_height / 2
            
            pos[node] = (x_px, y_px)
        elif "pos" in data:
            x_m, y_m = data["pos"]
            x_px = x_m / resolution
            y_px = -y_m / resolution
            x_px += frame_width / 2
            y_px += frame_height / 2
            pos[node] = (x_px, y_px)
        elif "pixel" in data:
            pos[node] = data["pixel"]
    
    # Debug: print coordinate ranges
    if pos and frame_idx % 1 == 0:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        print(f"[DEBUG Frame {frame_idx}] X range: {min(x_coords):.1f} to {max(x_coords):.1f} px")
        print(f"[DEBUG Frame {frame_idx}] Y range: {min(y_coords):.1f} to {max(y_coords):.1f} px")
        if base_image is not None:
            print(f"[DEBUG Frame {frame_idx}] Image shape: {base_image.shape} (H={base_image.shape[0]}, W={base_image.shape[1]})")
    
    
    # Draw edges with color coding
    known_edges = []
    new_edges = []
    
    for u, v in current_graph.edges():
        if u in known_nodes and v in known_nodes:
            known_edges.append((u, v))
        else:
            new_edges.append((u, v))
    
    # Draw known edges (magenta)
    if known_edges and pos:
        for u, v in known_edges:
            if u in pos and v in pos:
                x_coords = [pos[u][0], pos[v][0]]
                y_coords = [pos[u][1], pos[v][1]]
                ax.plot(x_coords, y_coords, 'm-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Draw new edges (blue)
    if new_edges and pos:
        for u, v in new_edges:
            if u in pos and v in pos:
                x_coords = [pos[u][0], pos[v][0]]
                y_coords = [pos[u][1], pos[v][1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2.0, alpha=0.8, zorder=2)
    
    # Draw known nodes (magenta)
    if known_nodes and pos:
        known_x = [pos[n][0] for n in known_nodes if n in pos]
        known_y = [pos[n][1] for n in known_nodes if n in pos]
        if known_x:
            ax.scatter(known_x, known_y, c='magenta', s=50, alpha=0.8, 
                      edgecolors='black', linewidths=0.5, zorder=3, label='Known nodes')
    
    # Draw new nodes (red)
    if new_nodes and pos:
        new_x = [pos[n][0] for n in new_nodes if n in pos]
        new_y = [pos[n][1] for n in new_nodes if n in pos]
        if new_x:
            ax.scatter(new_x, new_y, c='red', s=80, alpha=0.9, 
                      edgecolors='black', linewidths=1.0, zorder=4, label='New nodes')
    
    # Add shading for boundary nodes if available
    boundary_nodes = [n for n, data in current_graph.nodes(data=True) 
                     if data.get('node_type') == 'boundary']
    if boundary_nodes and pos:
        boundary_x = [pos[n][0] for n in boundary_nodes if n in pos]
        boundary_y = [pos[n][1] for n in boundary_nodes if n in pos]
        if boundary_x:
            ax.scatter(boundary_x, boundary_y, c='yellow', s=30, alpha=0.3, zorder=0)
    
    ax.set_title(f"Frame {frame_idx}: {len(current_graph.nodes())} nodes "
                f"({len(new_nodes)} new, {len(known_nodes)} known)")
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('equal')
    
    if base_image is not None:
        ax.set_xlim(0, base_image.shape[1])
        ax.set_ylim(base_image.shape[0], 0)
    
    plt.tight_layout()
    
    if save_path:
        # Save to file in headless mode
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[DEBUG] Saved visualization frame → {save_path}")
    else:
        # Interactive display
        plt.draw()
        plt.pause(0.5)  # Pause to allow display update


# ============================================================
# Main Builder
# ============================================================

def main():

    # ----------------------------------------------
    # 0. Clean up old results to keep only latest
    # ----------------------------------------------
    output_dir = "results"
    maps_dir = "results/simulated_route"
    
    # Remove old simulated_route if it exists
    if os.path.exists(maps_dir):
        import shutil
        shutil.rmtree(maps_dir)
        print(f"[INFO] Cleaned up old simulated route")
    
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------
    # 1. Setup parameters for simulation
    # ----------------------------------------------
    num_frames = 5
    crop_width = 256
    crop_height = 256

    # ----------------------------------------------
    # 1b. Calculate actual valid frames based on map size
    # ----------------------------------------------
    base_map_path = "../maps/carter_warehouse_navigation.png"
    temp_img = cv2.imread(base_map_path, cv2.IMREAD_GRAYSCALE)
    if temp_img is None:
        raise FileNotFoundError(f"Could not load base map: {base_map_path}")
    
    map_H, map_W = temp_img.shape[:2]
    max_frames_x = (map_W - crop_width) // STEP_PIXELS + 1
    actual_frames = min(num_frames, max_frames_x)
    
    print(f"[INFO] Base map: {map_W}x{map_H}, requested {num_frames} frames, will generate {actual_frames} valid frames")

    # ----------------------------------------------
    # 2. Build a global occupancy canvas and load the full base image
    # ----------------------------------------------
    H, W = crop_height, crop_width
    # Use actual_frames to calculate the global map width
    global_w = W + (actual_frames - 1) * STEP_PIXELS
    global_occupancy = np.zeros((H, global_w), dtype=np.uint8)
    global_map_path = os.path.join(output_dir, "global_map.png")
    out_overlay = os.path.join(output_dir, "global_graph_overlay.png")
    
    # Load the FULL base map for visualization background
    full_base_map = cv2.imread(base_map_path, cv2.IMREAD_GRAYSCALE)
    if full_base_map is None:
        print("[WARN] Could not load full base map for visualization")
        full_base_map = global_occupancy
    else:
        # Crop the base map to match the area being explored (using actual_frames)
        explore_width = (actual_frames - 1) * STEP_PIXELS + crop_width
        full_base_map = full_base_map[:crop_height, :explore_width]
        print(f"[INFO] Cropped base map to exploration area: {full_base_map.shape}")

    # ----------------------------------------------
    # 3. Configure SWAGGER generation
    # ----------------------------------------------
    # Use the main WaypointGraphGenerator (not the V2 variant)
    ConfigClass = WaypointGraphGeneratorConfig
    GeneratorClass = WaypointGraphGenerator

    config = ConfigClass(
        skeleton_sample_distance=0.20,
        boundary_inflation_factor=2,
        boundary_sample_distance=0.30,
        free_space_sampling_threshold=0.50,
        merge_node_distance=0.30,
        min_subgraph_length=0.35,
        use_skeleton_graph=False,
        use_boundary_sampling=True,
        use_free_space_sampling=True,
        use_delaunay_shortcuts=True,
        prune_graph=True,
        debug=True,  # Enable debug output for this example
    )

    generator = GeneratorClass(config)
    global_graph = None
    persisted_known_points: list[tuple[float, float]] = list(getattr(generator, "_known_points", []))
    
    # Use a merge distance that matches the graph generator's merge_node_distance
    # This prevents duplicate nodes in overlapping regions between frames
    global_builder = GlobalGraphGenerator(
        merge_distance=0.30,  # Match the generator's merge_node_distance
        retention_factor=1.0  # Keep all nodes (don't decay unseen nodes)
    )

    # ----------------------------------------------
    # Setup live visualization if enabled
    # ----------------------------------------------
    fig, ax = None, None
    is_headless = matplotlib.get_backend() == 'Agg'
    
    if ENABLE_LIVE_VISUALIZATION:
        if is_headless:
            print("[INFO] Headless mode detected - saving visualization frames instead of displaying")
            os.makedirs("results/viz_frames", exist_ok=True)
        else:
            plt.ion()  # Turn on interactive mode only if not headless
            print("[INFO] Live visualization enabled")
        
        fig, ax = plt.subplots(figsize=(12, 8))

    previous_global_graph = None

    # ----------------------------------------------
    # 4. Build + merge graph for each frame (generate on-the-fly)
    # ----------------------------------------------
    frame_generator = generate_robot_frames(
        input_image_path="../maps/carter_warehouse_navigation.png",
        output_dir=maps_dir,
        num_frames=num_frames,
        crop_width=crop_width,
        crop_height=crop_height,
        step_size=STEP_PIXELS,
        motion_direction="x",
        rotation=0.0  # For Isaac Lab, you would get this from robot's actual yaw
    )
    
    for i, occupancy, img_path, x_offset_px, y_offset_px, rotation_rad in frame_generator:
        print(f"\n=== Processing frame {i} ===")
        # Place the occupancy tile using ACTUAL offsets from generator
        x_px = x_offset_px
        y_px = y_offset_px
        global_occupancy[y_px:y_px + H, x_px:x_px + W] = occupancy
        cv2.imwrite(global_map_path, global_occupancy)


        # Compute world offsets for this frame using ACTUAL pixel offsets
        # Use top-left corner of the crop for world offset
        dx_m = x_px * RESOLUTION
        dy_m = y_px * RESOLUTION

        print(f"[DEBUG OFFSETS] Frame {i}: x_offset={x_px}px, y_offset={y_px}px, rotation={rotation_rad:.3f}rad → dx_m={dx_m:.4f}m, dy_m={dy_m:.4f}m")

        # Will print the world coordinates of the first node after graph generation below

        # Known points accumulated so far, provided in world coordinates
        known_points_world = list(persisted_known_points)

        # ----------------------------------------------
        # 4A. Generate local graph from occupancy grid
        # ----------------------------------------------
        robot_pose = (dx_m, dy_m, rotation_rad)  # Pass actual rotation from generator
        build_kwargs = dict(
            image=occupancy,
            resolution=RESOLUTION,
            safety_distance=SAFE_DIST,
            occupancy_threshold=127,
        )
        known_arg = known_points_world if known_points_world else []

        if hasattr(generator, "build_local_and_global_graphs"):
            local_graph, global_graph = generator.build_local_and_global_graphs(
                robot_pose=robot_pose,
                known_points=known_arg,
                **build_kwargs,
            )
        else:
            local_graph = generator.build_graph_from_grid_map(
                x_offset=robot_pose[0],
                y_offset=robot_pose[1],
                rotation=robot_pose[2],
                known_points=known_arg,
                **build_kwargs,
            )
            human_readable_global = getattr(generator, "_global_graph", None)
            if isinstance(human_readable_global, nx.Graph) and human_readable_global.number_of_nodes() > 0:
                global_graph = human_readable_global
            else:
                global_graph = local_graph

        # Debug: print world coordinates of a few nodes for this frame
        sample_nodes = list(local_graph.nodes(data=True))[:5]
        print(f"[DEBUG] Frame {i} sample node world coords:")
        for n, data in sample_nodes:
            if 'world' in data:
                print(f"    Node {n}: world={data['world']}")
            elif 'pos' in data:
                print(f"    Node {n}: pos={data['pos']}")
            else:
                print(f"    Node {n}: no world/pos info")
        if local_graph is None:
            print("[WARN] Failed to generate graph for frame")
            continue


        print(f"local graph node number: {len(local_graph)}")
        # Print world coordinates of the first node for this frame
        first_node = next(iter(local_graph.nodes(data=True)), None)
        if first_node:
            n, data = first_node
            if 'world' in data:
                print(f"[DEBUG] Frame {i} FIRST NODE: {n} world={data['world']}")
            elif 'pos' in data:
                print(f"[DEBUG] Frame {i} FIRST NODE: {n} pos={data['pos']}")
            else:
                print(f"[DEBUG] Frame {i} FIRST NODE: {n} no world/pos info")

        local_graph = ensure_graph_pos_consistent(local_graph)
        global_builder.add_local_graph(local_graph)
        print(f"global graph node number: {len(global_builder.get_global_graph())}")

        # ----------------------------------------------
        # Live visualization of graph building
        # ----------------------------------------------
        if ENABLE_LIVE_VISUALIZATION and ax is not None:
            current_global = global_builder.get_global_graph()
            save_path = f"results/viz_frames/frame_{i:03d}.png" if is_headless else None
            
            # Use full base map for background, not the growing global_occupancy
            visualize_graph_incremental(
                current_graph=current_global,
                previous_graph=previous_global_graph,
                frame_idx=i,
                ax=ax,
                base_image=full_base_map,  # Show full map, not stitched
                resolution=RESOLUTION,
                save_path=save_path
            )
            
            previous_global_graph = current_global.copy()

        # ----------------------------------------------
        # Export latest global graph as JSON
        # ----------------------------------------------
        try:
            def _make_json_serializable(obj):
                """Recursively convert numpy types, tuples, and other non-JSON-serializable
                objects into native Python types (lists, floats, ints, strs).
                """
                if obj is None or isinstance(obj, (str, bool, int, float)):
                    return obj
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer, _np.floating)):
                        return obj.item()
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                except Exception:
                    pass
                if isinstance(obj, dict):
                    return {str(k): _make_json_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_make_json_serializable(v) for v in obj]
                if isinstance(obj, tuple):
                    return [_make_json_serializable(v) for v in obj]
                try:
                    return float(obj)
                except Exception:
                    return str(obj)

            # Get the current global graph
            current_global = global_builder.get_global_graph()
            
            # Reindex nodes to simple integer ids
            mapping = {}
            id_map = {}
            for idx, n in enumerate(current_global.nodes()):
                mapping[n] = idx
                data = current_global.nodes[n]
                entry = {
                    "orig_id": list(n) if isinstance(n, (tuple, list)) else n,
                    "world": _make_json_serializable(data.get("world")),
                    "pos": _make_json_serializable(data.get("pos")),
                    "node_type": data.get("node_type"),
                }
                id_map[idx] = entry

            G_export = nx.relabel_nodes(current_global, mapping, copy=True)
            
            # Save latest global graph JSON (overwrites each iteration)
            json_path = os.path.join(output_dir, "latest_global_graph.json")
            node_link = json_graph.node_link_data(G_export)
            serializable = _make_json_serializable(node_link)
            with open(json_path, "w") as jf:
                json.dump(serializable, jf, indent=2)

            # Save id_map (overwrites each iteration)
            idmap_path = os.path.join(output_dir, "latest_global_graph_id_map.json")
            with open(idmap_path, "w") as imf:
                json.dump(id_map, imf, indent=2)

            print(f"[OK] Updated latest global graph JSON → {json_path}")
        except Exception as e:
            print(f"[WARN] Failed to export JSON graph: {e}")

        # Some generator implementations may not expose `global_graph` as a public
        # attribute. Use getattr() with a fallback to the global builder to avoid
        # AttributeError on generators that don't provide it.

    # ----------------------------------------------
    # Generate final overlay after all frames processed
    # ----------------------------------------------
    if ENABLE_LIVE_VISUALIZATION and fig is not None:
        # Keep the last visualization visible
        print("[INFO] Generating final overlay...")
    
    current_global = global_builder.get_global_graph()
    if isinstance(current_global, nx.Graph) and current_global.number_of_nodes() > 0:
        cv2.imwrite(global_map_path, global_occupancy)
        overlay_path = os.path.join(output_dir, "global_graph_overlay.png")
        visualize_global_graph_on_image(
            current_global,
            base_image_path=global_map_path,
            output_path=overlay_path,
        )
        print(f"[OK] Global graph overlay saved → {overlay_path}")

    # ============================================================
    # 5. Final global graph saved in loop above
    # ============================================================

    print("[OK] Processing complete.")
    print(f"    - Simulated route frames: {maps_dir}/")
    print(f"    - Global graph overlay: {output_dir}/global_graph_overlay.png")
    print(f"    - Latest global graph JSON: {output_dir}/latest_global_graph.json")
    if is_headless and ENABLE_LIVE_VISUALIZATION:
        print(f"    - Visualization frames: results/viz_frames/")

    # Keep visualization window open if enabled (only in non-headless mode)
    if ENABLE_LIVE_VISUALIZATION and fig is not None and not is_headless:
        plt.ioff()  # Turn off interactive mode
        print("[INFO] Close the visualization window to continue...")
        plt.show()  # This will block until the window is closed

    # ============================================================
    # 6. Example route query using pixel → world conversion
    # ============================================================

    start_px = (50, 150)
    goal_px = (450, 150)

    start = Point(x=start_px[0] * RESOLUTION, y=start_px[1] * RESOLUTION, z=0.0)
    goal  = Point(x=goal_px[0] * RESOLUTION, y=goal_px[1] * RESOLUTION, z=0.0)

    route = generator.find_route(start, goal)
    if route:
        print(f"[OK] Found route with {len(route)} waypoints.")
    else:
        print("[ERR] No route found.")


if __name__ == "__main__":
    main()
