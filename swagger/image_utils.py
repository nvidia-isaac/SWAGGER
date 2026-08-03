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

"""
Image I/O and raster drawing helpers backed by Pillow and scikit-image.

SWAGGER deliberately avoids ``opencv-python``: its wheels bundle FFmpeg and
OpenSSL 1.1.1, which drag those CVEs into the container image even though
SWAGGER only ever reads and writes still occupancy maps. Pillow, scikit-image
and SciPy already ship as dependencies, so these helpers add no new packages.
"""

import io
import os
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.draw import disk, line

# Pillow modes whose samples are already 8 bits and that convert losslessly to
# 8-bit grayscale. Anything outside this set (``I;16``, ``I``, ``F``, ...) is a
# higher bit depth than an occupancy map is allowed to use.
_EIGHT_BIT_MODES = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA"})


class ImageDecodeError(ValueError):
    """Raised when image bytes cannot be turned into an 8-bit grayscale map."""


def _to_grayscale_u8(image: Image.Image) -> np.ndarray:
    """
    Convert a Pillow image to a 2-D ``uint8`` grayscale array.

    Pillow's ``convert("L")`` applies the same ITU-R BT.601 luma weights as
    OpenCV's ``COLOR_BGR2GRAY`` (0.299 R + 0.587 G + 0.114 B), so grayscale
    values match to within the rounding of the final coefficient.
    """
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def read_grayscale(path: os.PathLike | str) -> Optional[np.ndarray]:
    """
    Read an image from disk as a 2-D ``uint8`` grayscale array.

    Mirrors ``cv2.imread(path, cv2.IMREAD_GRAYSCALE)``, including its habit of
    returning ``None`` rather than raising when the path is missing or the
    bytes are not a decodable image. Higher bit depths are scaled down to 8
    bits, as OpenCV does.

    Args:
        path: Path to the image file.

    Returns:
        The grayscale image, or ``None`` if it could not be read.
    """
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode in ("I;16", "I;16B", "I;16L", "I"):
                # Scale the high byte down, matching OpenCV's 16 -> 8 bit read.
                return (np.asarray(image, dtype=np.uint32) >> 8).astype(np.uint8)
            if image.mode == "F":
                return np.clip(np.asarray(image, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
            return _to_grayscale_u8(image)
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def read_rgb(path: os.PathLike | str) -> Optional[np.ndarray]:
    """
    Read an image from disk as an ``(H, W, 3)`` ``uint8`` RGB array.

    Unlike ``cv2.imread``, channels come back in RGB rather than BGR order.

    Args:
        path: Path to the image file.

    Returns:
        The RGB image, or ``None`` if it could not be read.
    """
    try:
        with Image.open(path) as image:
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def decode_grayscale(data: bytes) -> np.ndarray:
    """
    Decode in-memory image bytes into a 2-D ``uint8`` grayscale array.

    Args:
        data: Encoded image bytes.

    Returns:
        The grayscale image.

    Raises:
        ImageDecodeError: If the bytes are not a decodable image, or carry more
            than 8 bits per sample.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.mode not in _EIGHT_BIT_MODES:
                raise ImageDecodeError("Map image must be 8-bit.")
            return _to_grayscale_u8(image)
    except ImageDecodeError:
        # ImageDecodeError subclasses ValueError, so it has to be re-raised
        # before the clause below relabels it as an undecodable image.
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        # Pillow raises ValueError for some malformed palette images, and
        # np.asarray can too; without this the caller would see a 500.
        raise ImageDecodeError("Invalid image.") from error


def write_image(path: os.PathLike | str, image: np.ndarray) -> bool:
    """
    Write a grayscale or RGB ``uint8`` image to disk.

    Mirrors ``cv2.imwrite``'s boolean return so callers can keep reporting a
    failed write instead of propagating an exception. Unlike OpenCV, channels
    are interpreted as RGB rather than BGR.

    Args:
        path: Destination path; the extension selects the format.
        image: 2-D grayscale or 3-D RGB array.

    Returns:
        ``True`` if the file was written, ``False`` otherwise.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    elif array.ndim == 3 and array.shape[2] == 4:
        mode = "RGBA"
    else:
        return False
    try:
        Image.fromarray(array, mode=mode).save(path)
    except (OSError, ValueError, KeyError):
        return False
    return True


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Expand a 2-D grayscale image into a contiguous 3-channel RGB image."""
    return np.repeat(np.asarray(image, dtype=np.uint8)[:, :, np.newaxis], 3, axis=2)


def draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color) -> None:
    """
    Draw a 1-pixel Bresenham line between two ``(row, col)`` points, in place.

    Endpoints outside the image are clipped away rather than raising.

    Args:
        image: 2-D or 3-D array to draw into.
        start: ``(row, col)`` of the first endpoint.
        end: ``(row, col)`` of the second endpoint.
        color: Scalar for 2-D images, or a per-channel sequence for 3-D images.
    """
    rows, cols = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
    height, width = image.shape[:2]
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    image[rows[inside], cols[inside]] = color


def draw_disk(image: np.ndarray, center: tuple[int, int], radius: int, color) -> None:
    """
    Draw a filled disk centered on a ``(row, col)`` point, in place.

    Args:
        image: 2-D or 3-D array to draw into.
        center: ``(row, col)`` of the disk center.
        radius: Radius in pixels.
        color: Scalar for 2-D images, or a per-channel sequence for 3-D images.
    """
    # OpenCV's filled circle covers every pixel within squared distance
    # ``radius ** 2`` inclusive, while ``skimage.draw.disk`` is exclusive; the
    # epsilon makes the two footprints identical.
    rows, cols = disk((int(center[0]), int(center[1])), radius + 1e-3, shape=image.shape[:2])
    image[rows, cols] = color
