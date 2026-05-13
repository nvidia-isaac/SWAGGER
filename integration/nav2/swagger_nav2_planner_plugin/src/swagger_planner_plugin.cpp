/*
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>

#include "swagger_interfaces/srv/generate_route.hpp"

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
  nav2_util::declare_parameter_if_not_declared(
      node_, name_ + ".route_service_name", rclcpp::ParameterValue("generate_route"));
  node_->get_parameter(name_ + ".route_service_name", route_service_name_);
  route_client_ = std::make_unique<nav2_util::ServiceClient<
      swagger_interfaces::srv::GenerateRoute, nav2_util::LifecycleNode::SharedPtr>>(
      route_service_name_, node_);

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
    const geometry_msgs::msg::PoseStamped& start, const geometry_msgs::msg::PoseStamped& goal,
    std::function<bool()> cancel_checker) {
  nav_msgs::msg::Path empty_global_path;
  if (cancel_checker && cancel_checker()) {
    RCLCPP_INFO(node_->get_logger(), "Route planning was canceled before request submission");
    return empty_global_path;
  }

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

  auto request = std::make_shared<swagger_interfaces::srv::GenerateRoute::Request>();
  request->start = start.pose.position;
  request->goal = goal.pose.position;

  try {
    if (cancel_checker && cancel_checker()) {
      RCLCPP_INFO(node_->get_logger(), "Route planning was canceled before service call");
      return empty_global_path;
    }
    if (!route_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_ERROR(
          node_->get_logger(), "Route service %s is not available", route_service_name_.c_str());
      return empty_global_path;
    }
    auto response = route_client_->invoke(request, std::chrono::seconds(10));
    if (response->success) {
      return response->path;
    }
    RCLCPP_ERROR(
        node_->get_logger(), "Failed to generate route from service %s: %s",
        route_service_name_.c_str(), response->message.c_str());
  } catch (const std::exception& ex) {
    RCLCPP_ERROR(
        node_->get_logger(), "Failed to call route service %s: %s", route_service_name_.c_str(),
        ex.what());
  }
  return empty_global_path;
}

}  // namespace swagger_nav2_planner_plugin

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(swagger_nav2_planner_plugin::SwaggerRoutePlanner, nav2_core::GlobalPlanner)
