# Integration with ROS2 navigation2

These components show example integration of SWAGGER with [ROS2 navigation2](https://docs.nav2.org/#) for path planning.

## High level design

Three components:

1. The original library SWAGGER python library
2. ROS2 python node, wrapping route finding API and providing service interfaces to interact with SWAGGER
3. nav2 global planner plugin, which uses the above service API to generate paths.


## Usage

1. Clone this repo into a ros2 workspace's source directory, e.g. `~/ros2_ws/src`

   ```bash
   git clone git@github.com:nvidia-isaac/SWAGGER.git
   ```

2. Deactivate from your virtual environment if you are in one, because ROS2 uses the system level Python interpreter.

3. Ensure your pip version is at least 23.0.0 and setuptools at least 70.0.0 and less than 80.0.0. If not,

    ```bash
    pip install --upgrade pip setuptools==79.0.0 packaging
    ```

4. Install [NAV2](https://docs.nav2.org/getting_started/index.html#installation). Replace `<ros2-distro>` in the steps with `humble`.

5. Install the SWAGGER python library

   ```bash
   pip install -e SWAGGER  # or adjust path accordingly
   ```

6. Compile the ROS2 packages with colcon, from the workspace root, i.e. `~/ros2_ws` for the example above.

    ```bash
    cd ..
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --packages-up-to swagger_nav2_bringup
    source install/setup.bash # of your shell's equivalent like zsh
    ```

#### Running SWAGGER with NAV2

1. Launch IsaacSim and open the scene as explained in [this section](../README.md). Do not hit "Play" yet.
2. Run navigation with SWAGGER planner

    ```bash
    ros2 launch swagger_nav2_bringup swagger_with_nav2.launch.py \
        map_yaml:=src/waypoint_graph_generator/integration/nav2/swagger_nav2_bringup/maps/carter_warehouse_navigation.yaml \
        swagger_planner_config:=src/waypoint_graph_generator/integration/nav2/swagger_nav2_bringup/params/swagger_nav2_config.yaml
    ```

    Look out for logs like these to make sure the SWAGGER plugin is properly loaded

    ```bash
    [planner_server]: Created global planner plugin GridBased of type swagger_nav2_planner_plugin::SwaggerRoutePlanner
    [planner_server]: Created client for service generate_route
    [planner_server]: SWAGGER Planner Plugin initialized
    [planner_server]: Planner Server has GridBased  planners available.
    ```

3. Now hit **Play** in IsaacSim. This will initiate publishing of needed sensors and transform topics.
4. Follow the steps in [Localize the Robot and Start Navigation](../README.md#localize-the-robot-and-start-navigation) to start navigation.
