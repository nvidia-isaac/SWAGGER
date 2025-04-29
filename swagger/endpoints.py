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

import enum
import http
import mimetypes
import os
from typing import List, Optional

import cv2
import fastapi
import fastapi_versioning
import numpy as np
import requests

import swagger.models as models
from swagger.graph_manager import GraphManager
from swagger.utils import networkx_to_csr_graph

VERSION = 1


class Endpoints(str, enum.Enum):
    """REST API endpoints."""

    # Endpoint to check the health status
    HEALTH = "/health"
    # Endpoint to generate graph with a map and required data
    GRAPH_GENERATE = "/graph/generate"
    # Endpoint to get the generated graph by map_id
    GRAPH = "/graph"
    # Endpoint to query nearest nodes on the map by map_id
    NEAREST_NODES = "/graph/nearest_nodes"
    # Endpoint to get the visualization of the graph on map by map_id
    VISUALIZE = "/graph/visualize"
    # Endpoint to find the shortest route from a start to a goal in the real world
    FIND_ROUTE = "/graph/route"


# Set up FastAPI app
app = fastapi.FastAPI(
    title="SWAGGER REST API",
    description="Waypoint graph generation on occupancy maps for route planning.",
)

# Create graph manager with visualization directory
graph_manager = GraphManager(visualization_dir="visualizations")


@app.get(Endpoints.HEALTH, response_model=models.Health, status_code=http.HTTPStatus.OK)
@fastapi_versioning.version(VERSION)
async def health():
    """Gets the current status of the service."""
    errors = graph_manager.errors()  # Use GraphManager's errors() method
    status = models.HealthStatus.WARNING if errors else models.HealthStatus.RUNNING
    return models.Health(status=status, errors=errors)


async def _load_map(map_file: fastapi.UploadFile, map_uri: str = None) -> bytes:
    """Loads a map from a file attachment or a URI."""
    if map_file:
        return await _load_map_from_file(map_file)
    if map_uri:
        return _load_map_from_uri(map_uri)
    raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Map file or URI is not provided.")


async def _load_map_from_file(map_file: fastapi.UploadFile) -> bytes:
    """Load map from uploaded file and validate it's an image."""
    if not mimetypes.guess_type(map_file.filename)[0].startswith("image/"):
        raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Map must be an image file.")
    return await map_file.read()


def _load_map_from_uri(map_uri: str) -> bytes:
    """Load map from URI and validate it's an image."""
    response = requests.get(url=map_uri, stream=True)
    if not mimetypes.inited:
        mimetypes.init()
    content_type = mimetypes.guess_type(url=map_uri)[0]
    if not content_type or not content_type.startswith("image/"):
        raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Map must be an image file.")
    return response.content


def _decode_image(contents: bytes, occupancy_threshold: int = None) -> np.ndarray:
    """Decodes a map image and applies threshold to it."""
    if occupancy_threshold is None:
        raise fastapi.HTTPException(
            status_code=http.HTTPStatus.BAD_REQUEST, detail="Occupancy threshold is not provided for occupancy map."
        )

    image_bytes = np.frombuffer(contents, dtype="uint8")
    if image_bytes.size == 0:
        raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Empty image.")
    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _validate_map_id(map_id: str) -> str:
    """
    Validate and clean map ID.

    Args:
        map_id: Map identifier to validate

    Returns:
        Cleaned map ID with leading/trailing whitespace removed

    Raises:
        HTTPException: If map ID is None, empty, or only whitespace
    """
    if not map_id or not map_id.strip():
        raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Map ID is empty.")
    return map_id.strip()


@app.post(
    Endpoints.GRAPH_GENERATE,
    status_code=http.HTTPStatus.CREATED,
    responses={
        int(http.HTTPStatus.CREATED): {"model": models.CsrGraph},
        int(http.HTTPStatus.BAD_REQUEST): {"description": "Body missing required data"},
    },
)
@fastapi_versioning.version(VERSION)
async def post_graph(
    *,
    map_file: fastapi.UploadFile = fastapi.File(
        None, description="Occupancy map image file to generate the graph from."
    ),
    map_uri: str = fastapi.Form(None, description="Occupancy map image URI to generate the graph from."),
    data: models.MapData = fastapi.Depends(models.MapData.as_form),
):
    """Creates or updates a graph using a map file and required data."""
    if data.map_id is None:
        raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="Map ID is not provided.")

    map_id = _validate_map_id(data.map_id)
    content = await _load_map(map_file, map_uri)
    image = _decode_image(content, data.occupancy_threshold)

    if data.resolution is None:
        raise fastapi.HTTPException(
            status_code=http.HTTPStatus.BAD_REQUEST, detail="Resolution is not provided for occupancy map."
        )

    nx_graph = graph_manager.create_graph(
        map_id=map_id,
        image=image,
        resolution=data.resolution,
        safety_distance=data.safety_distance,
        occupancy_threshold=data.occupancy_threshold,
        x_offset=data.x_offset,
        y_offset=data.y_offset,
        rotation=data.rotation,
    )
    return networkx_to_csr_graph(nx_graph)


@app.get(
    Endpoints.GRAPH,
    responses={
        int(http.HTTPStatus.OK): {"model": models.CsrGraph},
        int(http.HTTPStatus.NOT_FOUND): {"description": "Graph not found"},
        int(http.HTTPStatus.BAD_REQUEST): {"description": "Invalid map ID"},
    },
)
@fastapi_versioning.version(VERSION)
async def get_graph(map_id: str):
    """Gets the graph of the map by its ID."""
    map_id = _validate_map_id(map_id)
    graph = graph_manager.get_graph(map_id)
    if graph is None:
        raise fastapi.HTTPException(
            status_code=http.HTTPStatus.NOT_FOUND, detail=f"Graph not found for map_id: {map_id}"
        )
    return networkx_to_csr_graph(graph)


@app.post(
    Endpoints.NEAREST_NODES,
    responses={
        int(http.HTTPStatus.OK): {"model": List[Optional[int]]},
        int(http.HTTPStatus.NOT_FOUND): {"description": "Graph not found"},
        int(http.HTTPStatus.BAD_REQUEST): {"description": "Invalid map ID"},
    },
)
@fastapi_versioning.version(VERSION)
async def get_nearest_node(point_list: models.PointList, map_id: str):
    """Gets the ID of the nearest waypoint nodes to a list of points."""
    map_id = _validate_map_id(map_id)
    try:
        return graph_manager.get_nearest_nodes(map_id, point_list.points)
    except KeyError as e:
        raise fastapi.HTTPException(status_code=http.HTTPStatus.NOT_FOUND, detail=str(e))


@app.get(
    Endpoints.VISUALIZE,
    responses={
        int(http.HTTPStatus.OK): {"content": {"image/png": {}}},
        int(http.HTTPStatus.NOT_FOUND): {"description": "Graph not found"},
        int(http.HTTPStatus.BAD_REQUEST): {"description": "Invalid map ID"},
    },
)
@fastapi_versioning.version(VERSION)
async def get_graph_image(map_id: str):
    """Gets the graph visualization image (PNG)."""
    map_id = _validate_map_id(map_id)
    success = graph_manager.visualize_graph(map_id)
    if not success:
        raise fastapi.HTTPException(
            status_code=http.HTTPStatus.NOT_FOUND, detail=f"Graph not found for map_id: {map_id}"
        )

    # Read and return the generated image
    image_path = graph_manager.get_visualization_path(map_id)
    if not os.path.exists(image_path):
        raise fastapi.HTTPException(
            status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to generate graph visualization"
        )

    with open(image_path, "rb") as f:
        return fastapi.responses.Response(content=f.read(), media_type="image/png")


@app.post(
    Endpoints.FIND_ROUTE,
    responses={
        int(http.HTTPStatus.OK): {"model": List[models.Point]},
        int(http.HTTPStatus.NOT_FOUND): {"description": "Graph not found"},
        int(http.HTTPStatus.BAD_REQUEST): {"description": "Invalid map ID"},
    },
)
@fastapi_versioning.version(VERSION)
async def post_route(start: models.Point, goal: models.Point, map_id: str):
    """Gets the shortest route from a start to a goal in the real world (POST method)."""
    map_id = _validate_map_id(map_id)
    try:
        route = graph_manager.find_route(map_id, start, goal)
        return route
    except KeyError as e:
        raise fastapi.HTTPException(status_code=http.HTTPStatus.NOT_FOUND, detail=str(e))


def get_versioned_app():
    return fastapi_versioning.VersionedFastAPI(app, version_format="{major}", prefix_format="/v{major}")
