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
from typing import Dict, List

import fastapi
import numpy as np
import pydantic

REGEX_FLOAT = r"[+-]?((\d+\.?\d*)|(\.\d+))"
REGEX_VECTOR2D = r",".join([REGEX_FLOAT] * 2)
REGEX_VECTOR3D = r",".join([REGEX_FLOAT] * 3)
REGEX_VECTOR4D = r",".join([REGEX_FLOAT] * 4)
REGEX_TILE_ID = r"^[+-]?\d*,[+-]?\d*,[+-]?\d*$"
REGEX_POINT2D = r"\(" + REGEX_VECTOR2D + r"\)"
REGEX_POLYGON2D = r"\(" + REGEX_POINT2D + r"(," + REGEX_POINT2D + r")*\)"


class HealthStatus(str, enum.Enum):
    """Waypoint graph generator REST API health status enums."""

    RUNNING = "running"
    WARNING = "warning"


class Health(pydantic.BaseModel):
    """Health status and errors of the service."""

    status: HealthStatus = pydantic.Field(..., description="REST API status")
    errors: Dict[str, str] = pydantic.Field(..., description="REST API errors")

    class Config:
        use_enum_values = True


class MapData(pydantic.BaseModel):
    """Metadata for graph generation."""

    map_id: str = pydantic.Field(..., description="Map ID to generate the graph from.")
    safety_distance: float = pydantic.Field(
        ..., description="Safety distance in meters to maintain from obstacles.", gt=0.0
    )
    resolution: float = pydantic.Field(..., description="Map resolution in meters per pixel.", gt=0.0)
    occupancy_threshold: int = pydantic.Field(
        ..., description="Threshold for determining occupied cells (0-255).", ge=0, le=255
    )
    x_offset: float = pydantic.Field(0.0, description="X offset of map origin in world coordinates (meters).")
    y_offset: float = pydantic.Field(0.0, description="Y offset of map origin in world coordinates (meters).")
    rotation: float = pydantic.Field(0.0, description="Rotation of map about Z axis in radians.")

    # Add Form support for FastAPI
    @classmethod
    def as_form(
        cls,
        map_id: str = fastapi.Form(..., description="Map ID to generate the graph from."),
        safety_distance: float = fastapi.Form(
            ..., description="Safety distance in meters to maintain from obstacles.", gt=0.0
        ),
        resolution: float = fastapi.Form(..., description="Map resolution in meters per pixel.", gt=0.0),
        occupancy_threshold: int = fastapi.Form(
            ..., description="Threshold for determining occupied cells (0-255).", ge=0, le=255
        ),
        x_offset: float = fastapi.Form(0.0, description="X offset of map origin in world coordinates (meters)."),
        y_offset: float = fastapi.Form(0.0, description="Y offset of map origin in world coordinates (meters)."),
        rotation: float = fastapi.Form(0.0, description="Rotation of map about Z axis in radians."),
    ) -> "MapData":
        return cls(
            map_id=map_id,
            safety_distance=safety_distance,
            resolution=resolution,
            occupancy_threshold=occupancy_threshold,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )


class Point(pydantic.BaseModel):
    """Represents a 3D point."""

    x: float = pydantic.Field(description="X coordinate of a point")
    y: float = pydantic.Field(description="Y coordinate of a point")
    z: float = pydantic.Field(description="Z coordinate of a point", default=0.0)

    def to_nparray(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_nparray(arr: np.array):
        return Point(x=arr[0], y=arr[1], z=arr[2])


class PointList(pydantic.BaseModel):
    """Contains a list of points."""

    points: List[Point] = pydantic.Field(None, description="List of points in world frame")


class CsrGraph(pydantic.BaseModel):
    """Defines the CSR graph model."""

    offsets: List[int] = pydantic.Field([], description="Edge offsets")
    edges: List[int] = pydantic.Field([], description="Edges")
    weights: List[float] = pydantic.Field([], description="Edge weights")
    nodes: List[Point] = pydantic.Field([], description="Node coordinates in world frame")
