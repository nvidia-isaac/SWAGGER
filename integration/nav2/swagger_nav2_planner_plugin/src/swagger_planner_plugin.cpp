/*
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "swagger_nav2_planner_plugin/swagger_planner_plugin.hpp"

#include "swagger_interfaces/msg/route_request.hpp"
#include "swagger_interfaces/msg/route_result.hpp"

namespace swagger_nav2_planner_plugin {

void SwaggerRoutePlanner::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) {
  node_ = parent.lock();
  name_ = name;
  tf_ = tf;
  global_frame_ = costmap_ros->getGlobalFrameID();

  // Declare parameters
  node_->declare_parameter("route_service_name", rclcpp::ParameterValue("generate_route"));
  node_->get_parameter("route_service_name", route_service_name_);

  // Create publisher for route request
  route_request_publisher_ =
      node_->create_publisher<swagger_interfaces::msg::RouteRequest>("route_request", 10);

  // Create subscription for route result
  route_result_subscription_ = node_->create_subscription<swagger_interfaces::msg::RouteResult>(
      "route_result", 10,
      std::bind(&SwaggerRoutePlanner::routeResultCallback, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(), "SWAGGER Planner Plugin initialized");
}

void SwaggerRoutePlanner::cleanup() {
  RCLCPP_INFO(node_->get_logger(), "CleaningUp plugin %s of type NavfnPlanner", name_.c_str());
}

void SwaggerRoutePlanner::activate() {
  RCLCPP_INFO(node_->get_logger(), "Activating plugin %s of type NavfnPlanner", name_.c_str());
}

void SwaggerRoutePlanner::deactivate() {
  RCLCPP_INFO(node_->get_logger(), "Deactivating plugin %s of type NavfnPlanner", name_.c_str());
}

nav_msgs::msg::Path SwaggerRoutePlanner::createPlan(
    const geometry_msgs::msg::PoseStamped& start, const geometry_msgs::msg::PoseStamped& goal) {
  nav_msgs::msg::Path empty_global_path;

  // Checking if the goal and start state is in the global frame
  if (start.header.frame_id != global_frame_) {
    RCLCPP_ERROR(
        node_->get_logger(), "Planner will only except start position from %s frame",
        global_frame_.c_str());
    return empty_global_path;
  }
  if (goal.header.frame_id != global_frame_) {
    RCLCPP_INFO(
        node_->get_logger(), "Planner will only except goal position from %s frame",
        global_frame_.c_str());
    return empty_global_path;
  }

  // Make request and send it to the route service
  swagger_interfaces::msg::RouteRequest request;
  request.start = start.pose.position;
  request.goal = goal.pose.position;
  request.route_id = ++route_id_;
  route_request_publisher_->publish(request);

  if (request.route_id == route_id_) {
    return global_path_;
  }
  return empty_global_path;
}

void SwaggerRoutePlanner::routeResultCallback(
    const swagger_interfaces::msg::RouteResult::SharedPtr msg) {
  if (route_id_ == msg->route_id) {
    RCLCPP_DEBUG(node_->get_logger(), "Route result received for route id %d", route_id_);
  }
  if (msg->success) {
    global_path_ = msg->path;
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Failed to generate route for id %d", route_id_);
  }
}

}  // namespace swagger_nav2_planner_plugin

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(swagger_nav2_planner_plugin::SwaggerRoutePlanner, nav2_core::GlobalPlanner)
