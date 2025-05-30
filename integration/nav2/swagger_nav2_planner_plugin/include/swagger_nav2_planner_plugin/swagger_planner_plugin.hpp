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

#pragma once

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"

#include "swagger_interfaces/msg/route_request.hpp"
#include "swagger_interfaces/msg/route_result.hpp"

namespace swagger_nav2_planner_plugin {

class SwaggerRoutePlanner : public nav2_core::GlobalPlanner {
 public:
  SwaggerRoutePlanner() = default;
  ~SwaggerRoutePlanner() = default;

  void configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent, std::string name,
      std::shared_ptr<tf2_ros::Buffer> tf,
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;
  nav_msgs::msg::Path createPlan(
      const geometry_msgs::msg::PoseStamped& start,
      const geometry_msgs::msg::PoseStamped& goal) override;

  void routeResultCallback(const swagger_interfaces::msg::RouteResult::SharedPtr msg);

 protected:
  std::shared_ptr<tf2_ros::Buffer> tf_;
  nav2_util::LifecycleNode::SharedPtr node_;
  std::string global_frame_, name_;
  std::string route_service_name_;

  uint32_t route_id_{0};
  nav_msgs::msg::Path global_path_;

  rclcpp::Publisher<swagger_interfaces::msg::RouteRequest>::SharedPtr route_request_publisher_;
  rclcpp::Subscription<swagger_interfaces::msg::RouteResult>::SharedPtr route_result_subscription_;
};

}  // namespace swagger_nav2_planner_plugin
