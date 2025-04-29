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

from typing import Tuple

import networkx as nx

from swagger.models import CsrGraph, Point


def pixel_to_world(
    row: float, col: float, resolution: float, x_offset: float, y_offset: float, cos_rot: float, sin_rot: float
) -> Point:
    """
    Convert pixel coordinates to world coordinates with the given transform.

    Coordinate system mapping:
    - In image frame: origin at top-left, x-axis points down (rows), y-axis points right (columns)
    - In ROS world frame: origin at bottom-left, x-axis points right, y-axis points up

    Args:
        row: Row coordinate in image frame (down direction)
        col: Column coordinate in image frame (right direction)
        resolution: Meters per pixel in the map
        x_offset: Translation in the x direction (meters)
        y_offset: Translation in the y direction (meters). When using ROS coordinates,
                 this should include the image height in meters (image_height * resolution)
        cos_rot: Cosine of the rotation angle
        sin_rot: Sine of the rotation angle

    Returns:
        Point object with world coordinates
    """
    # Convert from image coordinates to ROS world coordinates
    # In ROS: origin at bottom-left, x right, y up
    # In image: origin at top-left, row down, col right
    world_x = col * resolution  # col → ROS x (left to right)
    world_y = -row * resolution  # Negate row for y-up

    # Apply rotation
    x_rot = world_x * cos_rot - world_y * sin_rot
    y_rot = world_x * sin_rot + world_y * cos_rot

    # Apply translation (y_offset includes image height in meters)
    x_world = x_rot + x_offset
    y_world = y_rot + y_offset

    return Point(x=x_world, y=y_world, z=0.0)


def world_to_pixel(
    point: Point, resolution: float, x_offset: float, y_offset: float, cos_rot: float, sin_rot: float
) -> Tuple[int, int]:
    """
    Convert world coordinates to pixel coordinates with the inverse transform.

    Coordinate system mapping:
    - In ROS world frame: origin at bottom-left, x-axis points right, y-axis points up
    - In image frame: origin at top-left, x-axis points down (rows), y-axis points right (columns)

    Args:
        point: Point object with world coordinates
        resolution: Meters per pixel in the map
        x_offset: Translation in the x direction (meters)
        y_offset: Translation in the y direction (meters). When using ROS coordinates,
                 this should include the image height in meters (image_height * resolution)
        cos_rot: Cosine of the rotation angle
        sin_rot: Sine of the rotation angle

    Returns:
        Tuple of (row, column) pixel coordinates in image frame
    """
    # Remove translation (y_offset includes image height in meters)
    x = point.x - x_offset
    y = point.y - y_offset

    # Remove rotation (use negative angle)
    x_unrot = x * cos_rot + y * sin_rot
    y_unrot = -x * sin_rot + y * cos_rot

    # Convert to pixels
    # In ROS: x is right, y is up from bottom-left origin
    # In image: col is right, row is down from top-left origin
    col = x_unrot / resolution  # ROS x → image col
    row = -y_unrot / resolution  # Negate for row down

    return round(row), round(col)  # Return (row, column) for image indexing


def networkx_to_csr_graph(graph: nx.Graph) -> CsrGraph:
    """
    Convert a NetworkX graph to a CSR graph.
    """
    if graph.number_of_nodes() == 0:
        return CsrGraph()
    scipy_csr_graph = nx.to_scipy_sparse_array(graph)
    return CsrGraph(
        nodes=[
            Point(x=world_coords[0], y=world_coords[1], z=world_coords[2])
            for _, world_coords in graph.nodes(data="world")
        ],
        edges=scipy_csr_graph.indices.tolist(),
        offsets=scipy_csr_graph.indptr.tolist(),
        weights=scipy_csr_graph.data.tolist(),
    )
