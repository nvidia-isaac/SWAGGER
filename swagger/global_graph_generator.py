"""Global Graph Assembler for SWAGGER.

This helper consumes per-frame/local graphs (such as the ones produced by
``WaypointGraphGenerator``) and incrementally builds a stitched, world-frame
networkx graph. It handles:

* merging nearby free-space nodes (ignores skeleton nodes)
* persisting/freeing nodes using a retention factor (0 → only current frame,
  1 → keep everything forever)
* maintaining a probabilistic heatmap of boundary observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Set
import math

import cv2
import numpy as np
import networkx as nx


@dataclass
class GlobalGraphGenerator:
    """Incrementally build a stitched global graph.

    Args:
        merge_distance: Maximum world-distance between two free-space nodes
            to consider them the "same" node when merging (meters).
        retention_factor: Controls how long unseen nodes stay around.
            0.0 → drop nodes immediately if not observed this frame.
            1.0 → keep nodes forever.
        boundary_increment: How much to increment the obstacle probability
            whenever a boundary node is re-observed.
        boundary_decay: How quickly unseen obstacle probabilities decay.
    """

    merge_distance: float = 0.05
    retention_factor: float = 0.5
    boundary_increment: float = 0.2
    boundary_decay: float = 0.9
    boundary_cell_size: float = 0.05

    global_graph: nx.Graph = field(default_factory=nx.Graph, init=False)
    _next_node_id: int = field(default=0, init=False)
    _node_usage: Dict[int, float] = field(default_factory=dict, init=False)
    _boundary_probs: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False)

    def add_local_graph(self, local_graph: nx.Graph) -> None:
        """Merge a local graph into the persistent global graph."""

        if not 0.0 <= self.retention_factor <= 1.0:
            raise ValueError("retention_factor must be between 0 and 1")

        local_to_global: Dict = {}
        touched_nodes: Set[int] = set()
        seen_boundary_cells: Set[Tuple[int, int]] = set()

        for node, data in local_graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).lower()
            if node_type == "skeleton":
                continue  # never ingest skeleton nodes

            world = data.get("world")
            if world is None:
                continue
            world_xy = (float(world[0]), float(world[1]))

            if node_type == "boundary":
                key = self._quantize(world_xy)
                seen_boundary_cells.add(key)
                prob = self._boundary_probs.get(key, 0.0)
                prob = min(1.0, prob + self.boundary_increment)
                self._boundary_probs[key] = prob
                continue

            if node_type not in {"free_space", "known"}:
                continue

            global_id = self._merge_or_add_node(world_xy, data)
            local_to_global[node] = global_id
            touched_nodes.add(global_id)

        # Add edges only between nodes mapped above
        local_edge_pairs: Set[Tuple[int, int]] = set()
        for src, dst, edata in local_graph.edges(data=True):
            g_src = local_to_global.get(src)
            g_dst = local_to_global.get(dst)
            if g_src is None or g_dst is None or g_src == g_dst:
                continue
            pair = tuple(sorted((g_src, g_dst)))
            if pair in local_edge_pairs:
                continue  # keep only one connection per pair from this local graph
            local_edge_pairs.add(pair)

            if self.global_graph.has_edge(*pair):
                n1 = self.global_graph.nodes[g_src]["world"]
                n2 = self.global_graph.nodes[g_dst]["world"]
                dist = math.hypot(float(n1[0]) - float(n2[0]), float(n1[1]) - float(n2[1]))
                self.global_graph[g_src][g_dst]["weight"] = dist
                continue

            n1 = self.global_graph.nodes[g_src]["world"]
            n2 = self.global_graph.nodes[g_dst]["world"]
            dist = math.hypot(float(n1[0]) - float(n2[0]), float(n1[1]) - float(n2[1]))
            self.global_graph.add_edge(g_src, g_dst, weight=dist)

        # Clean up redundant edges after merging
        self._prune_redundant_edges()
        
        self._decay_nodes(touched_nodes)
        self._decay_boundaries(seen_boundary_cells)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_or_add_node(self, world_xy: Tuple[float, float], attrs: dict) -> int:
        """Merge with an existing node or create a new one."""
        best_node = None
        best_dist = self.merge_distance
        for node_id, data in self.global_graph.nodes(data=True):
            existing_world = data.get("world")
            if existing_world is None:
                continue
            dist = math.hypot(world_xy[0] - float(existing_world[0]), world_xy[1] - float(existing_world[1]))
            if dist < best_dist:
                best_dist = dist
                best_node = node_id

        if best_node is None:
            node_id = self._next_node_id
            self._next_node_id += 1
            z = 0.0
            if isinstance(attrs.get("world"), (tuple, list)) and len(attrs["world"]) >= 3:
                z = float(attrs["world"][2])
            node_data = {
                "world": (world_xy[0], world_xy[1], z),
                "node_type": "free_space",
                "origin": attrs.get("origin", "new"),
            }
            self.global_graph.add_node(node_id, **node_data)
            self._node_usage[node_id] = 1.0
            return node_id

        # Merge into existing node
        data = self.global_graph.nodes[best_node]
        existing_world = data.get("world", (world_xy[0], world_xy[1], 0.0))
        data["world"] = (world_xy[0], world_xy[1], existing_world[2])
        data["origin"] = attrs.get("origin", data.get("origin", "new"))
        self._node_usage[best_node] = 1.0
        return best_node

    def _decay_nodes(self, touched: Set[int]) -> None:
        threshold = 1e-3
        to_remove = []
        for node_id in list(self._node_usage.keys()):
            if node_id in touched:
                self._node_usage[node_id] = 1.0
                continue
            self._node_usage[node_id] *= self.retention_factor
            if self._node_usage[node_id] < threshold:
                to_remove.append(node_id)
        for node_id in to_remove:
            self.global_graph.remove_node(node_id)
            self._node_usage.pop(node_id, None)

    def _prune_redundant_edges(self) -> None:
        """Limit node degree by keeping only the closest neighbors."""
        max_degree = 8  # Reduced back to 8 for cleaner graph
        
        # Only prune every N frames to reduce overhead
        if not hasattr(self, '_prune_counter'):
            self._prune_counter = 0
        self._prune_counter += 1
        
        # Only run every 10 frames (less frequent)
        if self._prune_counter % 10 != 0:
            return
        
        edges_to_remove = set()  # Use set to avoid duplicates
        
        for node in self.global_graph.nodes():
            neighbors = list(self.global_graph.neighbors(node))
            degree = len(neighbors)
            
            # Only prune if extremely over-connected
            if degree <= max_degree:
                continue
            
            # Use list comprehension for efficiency
            neighbor_weights = [(self.global_graph[node][neighbor]["weight"], neighbor) 
                              for neighbor in neighbors]
            
            neighbor_weights.sort()  # Sort by distance (shortest first)
            
            # Keep only the max_degree closest neighbors
            for weight, neighbor in neighbor_weights[max_degree:]:
                edges_to_remove.add(tuple(sorted((node, neighbor))))
        
        # Batch remove edges
        self.global_graph.remove_edges_from(edges_to_remove)

    def _decay_boundaries(self, seen: Set[Tuple[int, int]]) -> None:
        for key in list(self._boundary_probs.keys()):
            if key in seen:
                continue
            self._boundary_probs[key] *= self.boundary_decay
            if self._boundary_probs[key] < 1e-3:
                del self._boundary_probs[key]

    def _quantize(self, world_xy: Tuple[float, float]) -> Tuple[int, int]:
        cell = self.boundary_cell_size
        return (int(math.floor(world_xy[0] / cell)), int(math.floor(world_xy[1] / cell)))

    def _cell_center(self, key: Tuple[int, int]) -> Tuple[float, float]:
        cell = self.boundary_cell_size
        return (key[0] * cell + cell / 2.0, key[1] * cell + cell / 2.0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_global_graph(self) -> nx.Graph:
        """Return the stitched global graph."""
        return self.global_graph

    def boundary_probabilities(self) -> Dict[Tuple[int, int], float]:
        """Return the current obstacle probability map."""
        return dict(self._boundary_probs)

    def debug_visualize(self, path: str, scale: float = 100.0) -> None:
        """Render a simple 2D visualization of the global graph."""

        if len(self.global_graph) == 0:
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(path, blank)
            return

        xs, ys = [], []
        for _, data in self.global_graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                continue
            xs.append(float(world[0]))
            ys.append(float(world[1]))
        if not xs or not ys:
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(path, blank)
            return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(64, int((max_x - min_x) * scale) + 64)
        height = max(64, int((max_y - min_y) * scale) + 64)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        def to_px(wx: float, wy: float) -> Tuple[int, int]:
            x = int((wx - min_x) * scale) + 32
            y = int((max_y - wy) * scale) + 32
            return x, y

        # Draw boundary probability shading (probabilities updated via exponential decay)
        if self._boundary_probs:
            overlay = canvas.copy()
            for key, prob in self._boundary_probs.items():
                wx, wy = self._cell_center(key)
                x, y = to_px(wx, wy)
                radius = max(2, int(self.boundary_cell_size * scale * 0.8))
                clamped = max(0.0, min(1.0, prob))
                color_val = int(clamped * 255)
                color = (0, color_val, 255 - color_val)
                cv2.circle(overlay, (x, y), radius, color, -1)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

        # Draw edges
        for u, v in self.global_graph.edges():
            n1 = self.global_graph.nodes[u]
            n2 = self.global_graph.nodes[v]
            if "world" not in n1 or "world" not in n2:
                continue
            x1, y1 = to_px(float(n1["world"][0]), float(n1["world"][1]))
            x2, y2 = to_px(float(n2["world"][0]), float(n2["world"][1]))
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Draw nodes
        for _, data in self.global_graph.nodes(data=True):
            world = data.get("world")
            if world is None:
                continue
            x, y = to_px(float(world[0]), float(world[1]))
            origin = str(data.get("origin", "new"))
            color = (0, 0, 255) if origin == "known" else (255, 0, 0)
            cv2.circle(canvas, (x, y), 3, color, -1)

        cv2.imwrite(path, canvas)
