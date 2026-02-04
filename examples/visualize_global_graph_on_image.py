import cv2
import numpy as np
import os

def visualize_global_graph_on_image(global_graph, base_image_path, output_path="results/global_graph_overlay.png"):
    """
    Overlay a combined graph onto the given base map image.
    Draws edges in red and nodes in green.
    """

    # --- Load the base map image ---
    base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
    if base_image is None:
        raise FileNotFoundError(f"[ERROR] Could not load base image: {base_image_path}")

    # Create a copy to draw on
    vis = base_image.copy()
    H, W, _ = vis.shape
    resolution = 0.05
    x_offset_m = 0.0
    y_offset_m = 0.0

    # --- Get all coordinates ---
    xs = []
    ys = []
    for _, n in global_graph.nodes(data=True):
        world = n.get("world", [0.0, 0.0, 0.0])
        xs.append(float(world[0]))
        ys.append(float(world[1]))

    # --- Auto scale factor ---
    x_scale = W / (max(xs) - min(xs) + 1e-6)
    y_scale = H / (max(ys) - min(ys) + 1e-6)
    scale = min(x_scale, y_scale)

    # --- Helper: convert world (m) → image (px) coordinates ---
    def world_to_pixel(x_m, y_m):
        x_px = int((x_m - min(xs)) * scale)
        y_px = int((y_m - min(ys)) * scale)
        return x_px, H - y_px  # flip Y so it’s upright

    # --- Draw edges ---
    for u, v, edge_data in global_graph.edges(data=True):
        node_u = global_graph.nodes[u]
        node_v = global_graph.nodes[v]

        wu = node_u.get("world", [0.0, 0.0, 0.0])
        wv = node_v.get("world", [0.0, 0.0, 0.0])
        x1, y1 = world_to_pixel(wu[0], wu[1])
        x2, y2 = world_to_pixel(wv[0], wv[1])
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Draw nodes ---
    for node_id, node_data in global_graph.nodes(data=True):
        world = node_data.get("world", [0.0, 0.0, 0.0])
        x, y = world_to_pixel(world[0], world[1])
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)  # Green nodes

    # --- Save the result ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"Global graph visualization saved to: {output_path}")
