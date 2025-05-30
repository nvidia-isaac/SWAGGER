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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="True",
        description="Use simulation clock if true",
    )
    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [get_package_share_directory("compass_swagger_navigator"), "params", "params.yaml"]
        ),
        description="Params file path",
    )
    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level to use for ROS nodes (debug, info, warn, error, fatal)",
    )
    # Fetch the args
    log_level = LaunchConfiguration("log_level")
    params_file = LaunchConfiguration("params_file")
    sim_time_arg = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration(
        "map_file",
        default=PathJoinSubstitution(
            [get_package_share_directory("swagger_nav2_bringup"), "maps", "carter_warehouse_navigation.png"]
        ),
    )

    nav_node = Node(
        name="compass_swagger_navigator",
        package="compass_swagger_navigator",
        executable="navigator",
        parameters=[
            {
                "policy_path": LaunchConfiguration("compass_policy_path"),
                "occupancy_map_file": map_file,
                "nav_command_topic": "cmd_vel",
                "image_topic": "/front_stereo_camera/left/image_raw",
                "goal_topic": "/goal_pose",
                "robot_frame": "base_link",
                "use_sim_time": sim_time_arg,
                "x_offset": -11.975,
                "y_offset": -17.975,
            }
        ],
        output="screen",
        arguments=["--ros-args", "--log-level", log_level, "--log-level", "rcl:=info"],
    )

    # Include nav2_bringup localization launch file
    nav2_bringup_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([get_package_share_directory("nav2_bringup"), "launch", "localization_launch.py"]),
            ]
        ),
        launch_arguments={
            "map": PathJoinSubstitution(
                [get_package_share_directory("swagger_nav2_bringup"), "maps", "carter_warehouse_navigation.yaml"]
            ),
            "params_file": params_file,
            "autostart": "true",
        }.items(),
    )

    # Include rviz launch file
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution([get_package_share_directory("swagger_nav2_bringup"), "launch", "rviz.launch.py"]),
            ]
        ),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(nav_node)
    ld.add_action(nav2_bringup_localization_launch)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(rviz_launch)
    return ld
