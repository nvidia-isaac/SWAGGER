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

import inspect
import logging
import os

import colorlog


class Logger:
    """A configurable logger class that provides standardized logging functionality."""

    # Define color scheme for different log levels
    COLORS = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize logger with specified name and level.

        Args:
            name: Logger name (typically __name__ of the calling module)
            level: Logging level (default: logging.INFO)
        """
        self._logger = logging.getLogger(name)
        if not self._logger.hasHandlers():
            self._configure_logger(level)

    def _configure_logger(self, level: int) -> None:
        """
        Configure logger with formatter and handler.

        Args:
            level: Logging level to set
        """
        formatter = colorlog.ColoredFormatter(
            "%(asctime)s.%(msecs)03d - %(log_color)s%(levelname)s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=self.COLORS,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(level)

    def _log(self, level: int, message: str) -> None:
        """Helper method to log messages with file and line info."""
        frame = inspect.currentframe().f_back.f_back
        file_name = os.path.basename(frame.f_code.co_filename)
        line_number = frame.f_lineno
        self._logger.log(level, f"{file_name}:{line_number} - {message}")

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._log(logging.INFO, message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._log(logging.ERROR, message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message)

    def set_level(self, level: int) -> None:
        """
        Set logging level.

        Args:
            level: New logging level
        """
        self._logger.setLevel(level)

    @property
    def level(self) -> int:
        """Get current logging level."""
        return self._logger.level
