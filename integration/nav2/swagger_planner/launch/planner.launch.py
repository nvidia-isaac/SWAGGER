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

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.substitutions.substitution_failure import SubstitutionFailure
from launch_ros.actions import Node


def load_yaml_if_exists(context):
    """Load parameters from YAML file if it exists."""
    try:
        params_file = LaunchConfiguration("params_file").perform(context)
    except SubstitutionFailure:
        return []
    if params_file and os.path.isfile(params_file):
        return [params_file]
    return []


def create_node(context):
    params = [
        {
            "map_file": LaunchConfiguration("map_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }
    ]

    yaml_params = load_yaml_if_exists(context)
    if yaml_params:
        params.extend(yaml_params)

    return [
        Node(
            name="swagger_route_planner",
            package="swagger_planner",
            executable="swagger_planner",
            parameters=params,
            output="screen",
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level"), "--log-level", "rcl:=info"],
        )
    ]


def generate_launch_description():
    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value="",
        description="Optional YAML file containing node parameters",
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Use simulation clock if true",
    )
    declare_map_file_cmd = DeclareLaunchArgument(
        "map_file",
        description="Map file path",
    )
    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level to use for ROS nodes (debug, info, warn, error, fatal)",
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_file_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(OpaqueFunction(function=create_node))
    ld.add_action(declare_params_file_cmd)
    return ld
