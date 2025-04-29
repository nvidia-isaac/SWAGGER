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

import dataclasses
import enum
import logging
import os
import sys

import requests
import tyro

from swagger.logger import Logger


class LogLevel(str, enum.Enum):
    """Log level options."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclasses.dataclass
class ClientConfig:
    """Configuration for the SWAGGER test client."""

    # Map configuration
    # Path to map image file
    map_path: str
    # Map ID
    map_id: str = "test_map"

    # Robot and map parameters
    # Robot radius in meters
    safety_distance: float = 0.3
    # Map resolution in meters per pixel
    resolution: float = 0.05

    # API connection
    # API host
    host: str = "localhost"
    # API port
    port: int = 8000

    # Output configuration
    # Output directory
    output_dir: str = "."
    # Log level
    log_level: LogLevel = LogLevel.INFO


class WaypointGraphClient:
    """Client for interacting with the SWAGGER REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", logger: Logger = None):
        """Initialize the client with the base URL of the API.

        Args:
            base_url: Base URL of the API service
            logger: Logger instance for logging
        """
        self.base_url = base_url.rstrip("/")
        self.api_version = "v1"
        self.base_api_url = f"{self.base_url}/{self.api_version}"
        self.logger = logger or Logger(__name__)

    def check_health(self) -> dict:
        """Check the health status of the service.

        Returns:
            Dict containing health status and any errors
        """
        self.logger.info(f"Checking health at {self.base_api_url}/health")
        response = requests.get(f"{self.base_api_url}/health")
        response.raise_for_status()
        return response.json()

    def generate_graph(
        self,
        map_id: str,
        map_path: str,
        safety_distance: float,
        resolution: float,
        occupancy_threshold: int = 127,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        rotation: float = 0.0,
    ) -> dict:
        """Generate a waypoint graph from a map image.

        Args:
            map_id: Unique identifier for the map
            map_path: Path to the map image file
            safety_distance: Radius of the robot in meters
            resolution: Map resolution in meters per pixel
            occupancy_threshold: Threshold for determining occupied cells (0-255)
            x_offset: X offset of map origin in world coordinates (meters)
            y_offset: Y offset of map origin in world coordinates (meters)
            rotation: Rotation of map about Z axis in radians

        Returns:
            Dict containing the generated graph data
        """
        if not os.path.exists(map_path):
            self.logger.error(f"Map file not found: {map_path}")
            raise FileNotFoundError(f"Map file not found: {map_path}")

        self.logger.info(f"Generating graph for map '{map_id}' from {map_path}")
        self.logger.debug(
            f"Parameters: safety_distance={safety_distance}, resolution={resolution}, "
            f"occupancy_threshold={occupancy_threshold}, x_offset={x_offset}, "
            f"y_offset={y_offset}, rotation={rotation}"
        )

        with open(map_path, "rb") as map_file:
            files = {"map_file": (os.path.basename(map_path), map_file, "image/png")}
            data = {
                "map_id": map_id,
                "safety_distance": str(safety_distance),
                "resolution": str(resolution),
                "occupancy_threshold": str(occupancy_threshold),
                "x_offset": str(x_offset),
                "y_offset": str(y_offset),
                "rotation": str(rotation),
            }
            response = requests.post(f"{self.base_api_url}/graph/generate", files=files, data=data)
            response.raise_for_status()
            return response.json()

    def get_graph(self, map_id: str) -> dict:
        """Get the graph for a specific map.

        Args:
            map_id: Unique identifier for the map

        Returns:
            Dict containing the graph data
        """
        self.logger.info(f"Retrieving graph for map '{map_id}'")
        response = requests.get(f"{self.base_api_url}/graph", params={"map_id": map_id})
        response.raise_for_status()
        return response.json()

    def get_nearest_nodes(self, map_id: str, points: list[dict[str, float]]) -> list[int | None]:
        """Get the nearest nodes to a list of points.

        Args:
            map_id: Unique identifier for the map
            points: List of points, each with x, y, z coordinates

        Returns:
            List of node indices, one for each input point
        """
        self.logger.info(f"Finding nearest nodes for {len(points)} points on map '{map_id}'")
        self.logger.debug(f"Points: {points}")

        data = {"points": points}
        response = requests.post(
            f"{self.base_api_url}/graph/nearest_nodes",
            params={"map_id": map_id},
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def get_visualization(self, map_id: str, output_path: str) -> bool:
        """Get the visualization of the graph and save it to a file.

        Args:
            map_id: Unique identifier for the map
            output_path: Path to save the visualization image

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Getting visualization for map '{map_id}'")
        self.logger.debug(f"Saving to: {output_path}")

        response = requests.get(
            f"{self.base_api_url}/graph/visualize",
            params={"map_id": map_id},
            stream=True,
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True


def get_log_level(level: LogLevel) -> int:
    """Convert log level enum to logging module constant."""
    levels = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    return levels[level]


def main():
    """Run the test client."""
    # Parse command line arguments using tyro
    args = tyro.cli(ClientConfig)

    # Set up logger
    logger = Logger("test_client", level=get_log_level(args.log_level))
    logger.info("Starting SWAGGER test client")

    # Create client
    base_url = f"http://{args.host}:{args.port}"
    logger.info(f"Connecting to API at {base_url}")
    client = WaypointGraphClient(base_url, logger=logger)

    # Test health endpoint
    logger.info("Testing health endpoint...")
    try:
        health = client.check_health()
        logger.info(f"Health status: {health['status']}")
        if health["errors"]:
            logger.warning(f"Service reported errors: {health['errors']}")
    except requests.RequestException as e:
        logger.error(f"Error checking health: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Unexpected response format: missing key {e}")
        sys.exit(1)

    # Test graph generation
    logger.info("Testing graph generation...")
    try:
        graph = client.generate_graph(
            map_id=args.map_id,
            map_path=args.map_path,
            safety_distance=args.safety_distance,
            resolution=args.resolution,
        )
        logger.info(f"Generated graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    except FileNotFoundError as e:
        logger.error(f"Map file error: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        logger.error(f"API request error during graph generation: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Unexpected response format: missing key {e}")
        sys.exit(1)

    # Test get graph
    logger.info("Testing get graph...")
    try:
        graph = client.get_graph(args.map_id)
        logger.info(f"Retrieved graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    except requests.RequestException as e:
        logger.error(f"API request error retrieving graph: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Unexpected response format: missing key {e}")
        sys.exit(1)

    # Test nearest nodes
    logger.info("Testing nearest nodes...")
    try:
        # Use the first node as a test point
        if graph["nodes"]:
            test_points = [
                {"x": graph["nodes"][0]["x"], "y": graph["nodes"][0]["y"], "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.0},  # Origin
            ]
            nearest_nodes = client.get_nearest_nodes(args.map_id, test_points)
            logger.info(f"Nearest nodes: {nearest_nodes}")
        else:
            logger.warning("No nodes in graph to test nearest_nodes endpoint")
    except requests.RequestException as e:
        logger.error(f"API request error finding nearest nodes: {e}")
        sys.exit(1)
    except (KeyError, IndexError) as e:
        logger.error(f"Error accessing data: {e}")
        sys.exit(1)

    # Test visualization
    logger.info("Testing visualization...")
    try:
        output_path = os.path.join(args.output_dir, f"{args.map_id}_graph.png")
        success = client.get_visualization(args.map_id, output_path)
        if success:
            logger.info(f"Visualization saved to {output_path}")
        else:
            logger.warning("Failed to get visualization")
    except requests.RequestException as e:
        logger.error(f"API request error getting visualization: {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Error saving visualization file: {e}")
        sys.exit(1)

    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    main()
