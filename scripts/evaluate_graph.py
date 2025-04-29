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

import json
import os
from dataclasses import dataclass

import cv2
import networkx as nx
import tyro

from swagger.graph_evaluator import WaypointGraphEvaluator


@dataclass
class Args:
    # Path to the input graph file in CSR format
    graph_path: str
    # Path to the occupancy grid map image
    map_path: str
    # Directory to save evaluation results
    output_dir: str
    # Map resolution in meters per pixel
    resolution: float
    # Robot radius in meters for collision checking
    safety_distance: float
    # Threshold value to determine free vs occupied space in occupancy map
    occupancy_threshold: int
    # X translation offset in meters
    x_offset: float = 0.0
    # Y translation offset in meters
    y_offset: float = 0.0
    # Rotation in radians
    rotation: float = 0.0


def main():
    args = tyro.cli(Args)
    nx_graph = nx.read_gml(args.graph_path)

    occupancy_map = cv2.imread(args.map_path, cv2.IMREAD_GRAYSCALE)
    evaluator = WaypointGraphEvaluator(
        graph=nx_graph,
        occupancy_map=occupancy_map,
        resolution=args.resolution,
        safety_distance=args.safety_distance,
        occupancy_threshold=args.occupancy_threshold,
    )
    results = evaluator.evaluate_all(print_metrics=True)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + "/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
