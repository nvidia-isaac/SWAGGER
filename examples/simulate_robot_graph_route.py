import os
import cv2
import time
import numpy as np
from pathlib import Path
from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig
from swagger.models import Point


def simulate_robot_graph_route(
    input_image_path: str,
    output_dir: str = "results/simulated_route",
    num_frames: int = 10,
    crop_width: int = 256,
    crop_height: int = 256,
    step_size: int = 100,
    motion_direction: str = "x",
):
    """
    Simulates a robot moving through a large environment by cropping submaps,
    generating graphs for each, and combining them into a global graph.

    Args:
        input_image_path (str): Path to the large occupancy map.
        output_dir (str): Directory to save results.
        num_frames (int): Number of cropped regions (steps).
        crop_width (int): Width of each cropped map (pixels).
        crop_height (int): Height of each cropped map (pixels).
        step_size (int): Pixel distance between consecutive crops.
        motion_direction (str): 'x', 'y', or 'xy' for movement direction.
        offset_distance (float): Distance offset between graphs (meters).
        visualize_route (bool): Whether to visualize the robot’s movement.

    Returns:
        generator (WaypointGraphGenerator): The graph generator object with all subgraphs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load base map
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {input_image_path}")
    H, W = image.shape[:2]
    print(f"[INFO] Loaded base map: {W}x{H}")

    # Iterate through simulated motion
    for i in range(num_frames):
        # Calculate crop region
        if motion_direction == "x":
            x_start, y_start = i * step_size, 0
        elif motion_direction == "y":
            x_start, y_start = 0, i * step_size
        elif motion_direction == "xy":
            x_start, y_start = i * step_size, i * step_size
        else:
            raise ValueError("motion_direction must be 'x', 'y', or 'xy'")

        x_start = min(x_start, W - crop_width)
        y_start = min(y_start, H - crop_height)
        x_offset = i * step_size * 0.039
        print(f"[DEBUG] Frame {i}: x_start={x_start}px, x_offset={x_offset:.3f}m")

        cropped = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
        crop_filename = os.path.join(output_dir, f"frame_{i+1:03d}.png")
        cv2.imwrite(crop_filename, cropped)

        print(f"[INFO] Saved cropped frame {i+1}/{num_frames} to {crop_filename}")

    print(f"\n[INFO] Generated {num_frames} cropped maps in '{output_dir}'")

