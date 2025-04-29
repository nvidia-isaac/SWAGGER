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

import math
import unittest

from swagger.models import Point
from swagger.utils import pixel_to_world, world_to_pixel


class TestCoordinateTransforms(unittest.TestCase):
    def test_identity_transform(self):
        """Test with identity transform (no rotation or translation)."""
        # Setup
        resolution = 0.1  # 10 cm per pixel
        x_offset = 0.0
        y_offset = 0.0
        cos_rot = 1.0  # cos(0)
        sin_rot = 0.0  # sin(0)

        # Test pixel_to_world
        row, col = 10, 20
        point = pixel_to_world(row, col, resolution, x_offset, y_offset, cos_rot, sin_rot)

        # With identity transform, x = col * resolution, y = -row * resolution
        self.assertAlmostEqual(point.x, col * resolution)
        self.assertAlmostEqual(point.y, -row * resolution)
        self.assertEqual(point.z, 0.0)

        # Test world_to_pixel (round-trip)
        new_row, new_col = world_to_pixel(point, resolution, x_offset, y_offset, cos_rot, sin_rot)
        self.assertEqual(new_row, row)
        self.assertEqual(new_col, col)

    def test_translation(self):
        """Test with translation only."""
        # Setup
        resolution = 0.05  # 5 cm per pixel
        x_offset = 10.0  # 10 meters offset in x
        y_offset = -5.0  # -5 meters offset in y
        cos_rot = 1.0
        sin_rot = 0.0

        # Test pixel_to_world
        row, col = 30, 40
        point = pixel_to_world(row, col, resolution, x_offset, y_offset, cos_rot, sin_rot)

        # Expected: x = row * resolution + x_offset, y = col * resolution + y_offset
        self.assertAlmostEqual(point.x, col * resolution + x_offset)
        self.assertAlmostEqual(point.y, -row * resolution + y_offset)

        # Test world_to_pixel (round-trip)
        new_row, new_col = world_to_pixel(point, resolution, x_offset, y_offset, cos_rot, sin_rot)
        self.assertEqual(new_row, row)
        self.assertEqual(new_col, col)

    def test_rotation(self):
        """Test with rotation only (90 degrees)."""
        # Setup
        resolution = 0.1
        x_offset = 0.0
        y_offset = 0.0
        angle = math.pi / 2  # 90 degrees
        cos_rot = math.cos(angle)
        sin_rot = math.sin(angle)

        # Test pixel_to_world
        row, col = 10, 20
        point = pixel_to_world(row, col, resolution, x_offset, y_offset, cos_rot, sin_rot)

        # With 90-degree rotation: x = row * resolution, y = col * resolution
        self.assertAlmostEqual(point.x, row * resolution, places=5)
        self.assertAlmostEqual(point.y, col * resolution, places=5)

        # Test world_to_pixel (round-trip)
        new_row, new_col = world_to_pixel(point, resolution, x_offset, y_offset, cos_rot, sin_rot)
        self.assertAlmostEqual(new_row, row, places=5)
        self.assertAlmostEqual(new_col, col, places=5)

    def test_combined_transform(self):
        """Test with both rotation and translation."""
        # Setup
        resolution = 0.05
        x_offset = 2.5
        y_offset = -1.5
        angle = math.pi / 4  # 45 degrees
        cos_rot = math.cos(angle)
        sin_rot = math.sin(angle)

        # Test pixel_to_world
        row, col = 15, 25
        point = pixel_to_world(row, col, resolution, x_offset, y_offset, cos_rot, sin_rot)

        # Calculate expected values manually
        x = row * resolution
        y = col * resolution
        x_rot = x * cos_rot + y * sin_rot
        y_rot = -x * sin_rot + y * cos_rot
        expected_x = x_rot + x_offset
        expected_y = y_rot + y_offset

        self.assertAlmostEqual(point.x, expected_x, places=5)
        self.assertAlmostEqual(point.y, expected_y, places=5)

        # Test world_to_pixel (round-trip)
        new_row, new_col = world_to_pixel(point, resolution, x_offset, y_offset, cos_rot, sin_rot)
        self.assertAlmostEqual(new_row, row, places=5)
        self.assertAlmostEqual(new_col, col, places=5)

    def test_rounding_in_world_to_pixel(self):
        """Test that world_to_pixel properly rounds to integers."""
        # Setup
        resolution = 0.1
        x_offset = 0.0
        y_offset = 0.0
        cos_rot = 1.0
        sin_rot = 0.0

        # Create a point that would result in non-integer pixel coordinates
        point = Point(x=2.34, y=5.67, z=0.0)

        # Test world_to_pixel
        row, col = world_to_pixel(point, resolution, x_offset, y_offset, cos_rot, sin_rot)

        # Expected: round(x/resolution), round(y/resolution)
        self.assertEqual(col, round(point.x / resolution))
        self.assertEqual(row, round(-point.y / resolution))
        self.assertTrue(isinstance(row, int))
        self.assertTrue(isinstance(col, int))


if __name__ == "__main__":
    unittest.main()
