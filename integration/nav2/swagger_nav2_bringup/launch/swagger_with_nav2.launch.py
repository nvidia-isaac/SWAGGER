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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="True",
        description="Use simulation clock if true",
    )
    declare_map_file_cmd = DeclareLaunchArgument(
        "map_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("swagger_nav2_bringup"), "maps", "carter_warehouse_navigation.png"]
        ),
        description="Map file path",
    )
    declare_map_yaml_cmd = DeclareLaunchArgument(
        "map_yaml",
        default_value=PathJoinSubstitution(
            [FindPackageShare("swagger_nav2_bringup"), "maps", "carter_warehouse_navigation.yaml"]
        ),
        description="Map yaml file path",
    )
    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level to use for ROS nodes (debug, info, warn, error, fatal)",
    )
    declare_swagger_planner_config_cmd = DeclareLaunchArgument(
        "swagger_planner_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("swagger_nav2_bringup"), "params", "swagger_nav2_config.yaml"]
        ),
        description="Path to the configuration file",
    )

    sim_time_arg = LaunchConfiguration("use_sim_time")
    map_file_arg = LaunchConfiguration("map_file")
    log_level_arg = LaunchConfiguration("log_level")
    swagger_planner_config_arg = LaunchConfiguration("swagger_planner_config")

    # Include swagger_planner launch file
    swagger_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([FindPackageShare("swagger_planner"), "launch", "planner.launch.py"]),
            ]
        ),
        launch_arguments={
            "use_sim_time": sim_time_arg,
            "map_file": map_file_arg,
            "log_level": log_level_arg,
            "params_file": swagger_planner_config_arg,
        }.items(),
    )

    # Include nav2_bringup navigation2 launch file
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"]),
            ]
        ),
        launch_arguments={
            "use_sim_time": sim_time_arg,
            "params_file": swagger_planner_config_arg,
        }.items(),
    )

    # Include nav2_bringup localization launch file
    nav2_bringup_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "localization_launch.py"]),
            ]
        ),
        launch_arguments={
            "map": LaunchConfiguration("map_yaml"),
            "params_file": swagger_planner_config_arg,
            "autostart": "true",
        }.items(),
    )

    # Include rviz launch file
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([FindPackageShare("swagger_nav2_bringup"), "launch", "rviz.launch.py"]),
            ]
        ),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_file_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(swagger_planner_launch)
    ld.add_action(nav2_bringup_launch)
    ld.add_action(nav2_bringup_localization_launch)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_swagger_planner_config_cmd)
    ld.add_action(rviz_launch)

    return ld
