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

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import psutil

from swagger.logger import Logger
from swagger.models import Point


@dataclass
class PerformanceResult:
    """Container for performance measurement results."""

    total_time: float  # Total execution time in seconds
    peak_memory: float  # Peak memory usage in MB
    breakdown: Dict[str, float]  # Timing breakdown by component


class PerformanceEvaluator:
    """Class to evaluate computational performance metrics."""

    def __init__(self):
        self._logger = Logger(__name__)
        self._start_time = None
        self._timings = {}
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss / 1024 / 1024  # Convert to MB
        self._peak_memory = self._initial_memory  # Initialize peak memory to initial memory

    def start(self, name: str = "total"):
        """Start timing a section of code."""
        if name == "total":
            self._start_time = time.time()
        self._timings[name] = {"start": time.time()}
        # Update memory usage at the start of each section
        self._update_peak_memory()

    def stop(self, name: str = "total") -> float:
        """Stop timing a section and return duration."""
        if name not in self._timings:
            return 0.0

        # Update memory usage at the end of each section
        self._update_peak_memory()

        duration = time.time() - self._timings[name]["start"]
        self._timings[name]["duration"] = duration
        return duration

    def _update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self._process.memory_info().rss / 1024 / 1024  # Convert to MB
        if current_memory > self._peak_memory:
            self._peak_memory = current_memory

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        self._update_peak_memory()  # Ensure we have the latest memory usage
        return self._peak_memory - self._initial_memory

    def evaluate_query_performance(self, graph_generator, num_queries: int = 100) -> Dict[str, float]:
        """
        Evaluate waypoint query performance.

        Args:
            graph_generator: WaypointGraphGenerator instance with built graph
            num_queries: Number of random queries to test

        Returns:
            Dictionary containing query performance metrics
        """
        self._logger.info(f"Evaluating query performance with {num_queries} random queries...")

        # Generate random query points
        height, width = graph_generator._original_map.shape
        random_points = []
        for _ in range(num_queries):
            x = np.random.uniform(0, width * graph_generator._resolution)
            y = np.random.uniform(0, height * graph_generator._resolution)
            random_points.append(Point(x=x, y=y, z=0.0))

        # Time the queries
        query_times = []
        self.start("queries")
        for point in random_points:
            start = time.time()
            graph_generator.get_node_ids([point])
            query_times.append(time.time() - start)
        self.stop("queries")

        return {
            "average_query_time": float(np.mean(query_times)),
            "max_query_time": float(np.max(query_times)),
            "min_query_time": float(np.min(query_times)),
            "total_query_time": float(np.sum(query_times)),
        }

    def get_results(self) -> PerformanceResult:
        """Get final performance results."""
        if "total" in self._timings:
            total_time = time.time() - self._start_time
        else:
            total_time = 0.0

        # Get timing breakdown
        breakdown = {
            name: timing["duration"]
            for name, timing in self._timings.items()
            if "duration" in timing and name != "total"
        }

        return PerformanceResult(total_time=total_time, peak_memory=self.get_peak_memory(), breakdown=breakdown)
