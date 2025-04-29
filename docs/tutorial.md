# Tutorial

In this tutorial, you will learn
* the required data for waypoint graph generation,
* how to use the SWAGGER library in your Python code, and
* how to use the REST API with a non-Python application.

This tutorial assumes that you have installed the SWAGGER library and its dependencies.

## Table of Contents
- [Python](#python)
  - [Prepare Data](#prepare-data)
  - [Build a Graph](#build-a-graph)
  - [Visualize Graph on Map](#visualize-graph-on-map)
  - [Find Route](#find-route)
  - [Nearest Node](#nearest-node)
  - [Tuning Graph Building Parameters](#tuning-graph-building-parameters)
  - [Manage Multiple Maps with Graph Manager](#manage-multiple-maps-with-graph-manager)
- [REST API](#rest-api)

## Python

The complete example can be found at [waypoint_graph_generation_example.py](../examples/waypoint_graph_generation_example.py).

### Prepare Data

```python
import cv2
from pathlib import Path

# The example script is in the `example` directory
occupancy_grid = cv2.imread(Path(__file__).parent.parent / "data" / "carter_warehouse_navigation.png", cv2.IMREAD_GRAYSCALE)
occupancy_threshold = 127
safety_distance = 0.3
resolution = 0.05
x_offset = 0.0
y_offset = 0.0
rotation = 0.0
```

These values are cached in the waypoint graph generator object after graph building for visualization and query purposes.

### Build a Graph

```python
from swagger import WaypointGraphGenerator

generator = WaypointGraphGenerator()
graph = generator.build_graph_from_grid_map(
    image=occupancy_grid,
    occupancy_threshold=occupancy_threshold,
    safety_distance=safety_distance,
    resolution=resolution,
    x_offset=x_offset,
    y_offset=y_offset,
    rotation=rotation,
)
```

The resulting graph is a `networkx.Graph` object. Each node has the following data:

* `world`: the node's coordinates in real world.
* `pixel`: the node's pixel on the occupancy map.
* `node_type`: the node's type, indicating the graph building stage when it was added to the graph. If the value is not set, it was added during the skeleton graph building stage.

Each edge has the following data:

* `weight`: the real-world Euclidean distance between the two nodes it connects.
* `edge_type`: the edge's type, indicating the graph building stage when it was added to the graph.

To inspect node and edge data,

```python
print(f"Node 1: {graph.nodes[1]}")
print(f"Edge (0, 1): {graph.edges[(0, 1)]}")
```

For more NetworkX operations, please refer to NetworkX's official documentation.

### Visualize Graph on Map

To create a visualization of the graph on top of the map image,

```python
import os

output_dir = "results"
output_filename = "waypoint_graph_example.png"
os.makedirs(output_dir, exist_ok=True)
generator.visualize_graph(output_dir=output_dir, output_filename=output_filename)
```

This saves the visualization as an image to your desired location with the specified file name.

### Find Route

To find the shortest route from a start to a goal in the real world,

```python
from swagger.models import Point

start = Point(x=13.0, y=18.0)
goal = Point(x=15.0, y=10.0)
route = generator.find_route(start, goal)
```

If there is a route between the two points, the returned `route` is a list of `Point`s. Otherwise, error logs will appear on the console and `route` will be an empty list.


### Nearest Node

To find the nearest nodes to a list of real-world points,

```python
points = [Point(x=13.0, y=18.0), Point(x=15.0, y=10.0), Point(x=0.0, y=0.0)]
node_ids = generator.get_node_ids(points)
```

This returns a list of IDs corresponding to the closest node each point can reach. If a point is in collision or cannot reach any node, its corresponding value will be `None`.

### Tuning Graph Building Parameters

The waypoint graph generator has several parameters that control how the graph is constructed. These parameters can be adjusted to optimize the graph for different environments and robot requirements.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `grid_sample_distance` | 1.5 m | Maximum distance between waypoints when creating a grid graph on a completely free map. Lower values create denser graphs with more nodes. |
| `skeleton_sample_distance` | 1.5 m | Maximum distance between waypoints when creating a skeleton graph. Lower values create denser graphs with more nodes. |
| `boundary_inflation_factor` | 1.5 | Factor to inflate boundaries for contour detection. Higher values create larger margins around obstacles. This is unitless. |
| `boundary_sample_distance` | 2.5 m | Distance between samples along obstacle contours. Lower values create more waypoints along boundaries. |
| `free_space_sampling_threshold` | 1.5 m | Maximum distance away from obstacles to sample free space. Controls how far from obstacles the free space samples are placed. |
| `merge_node_distance` | 0.25 m | Maximum distance between nodes to merge them. Helps reduce redundant nodes that are close to each other. |
| `min_subgraph_length` | 0.25 m | Minimum total edge length required to keep a subgraph. Helps eliminate disconnected or tiny subgraphs. |
| `use_skeleton_graph` | True | Whether to use skeleton graph generation, which creates a graph along the medial axis of free space. |
| `use_boundary_sampling` | True | Whether to add nodes along obstacle boundaries. Useful for navigation paths that need to traverse close to obstacles. |
| `use_free_space_sampling` | True | Whether to add nodes in open spaces. Helps ensure coverage in large open areas. |
| `use_delaunay_shortcuts` | True | Whether to add shortcut edges using Delaunay triangulation. Creates more direct paths between nodes. |
| `prune_graph` | True | Whether to prune the graph to remove redundant nodes and edges. Helps optimize the final graph for efficiency. |


You can set these parameters when creating the waypoint graph generator instance:

```python
from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig

# Create a custom configuration
config = WaypointGraphGeneratorConfig(
    waypoint_interval=1.0,               # Distance between waypoints (in meters)
    boundary_inflation_factor=2.0,       # Larger margins around obstacles
    boundary_sample_distance=1.5,        # Distance between boundary samples (in meters)
    free_space_sampling_threshold=2.0,   # Distance from obstacles for free space sampling (in meters)
    merge_node_distance=0.3,             # Distance to merge nearby nodes (in meters)
    min_subgraph_length=1.0,             # Minimum subgraph length to keep (in meters)
    use_skeleton_graph=True,             # Create a graph along the medial axis
    use_boundary_sampling=True,          # Sample nodes along boundaries
    use_free_space_sampling=True,        # Sample nodes in open areas
    use_delaunay_shortcuts=True,         # Create shortcut edges
    prune_graph=True                     # Remove redundant nodes and edges
)

# Create generator with custom config
generator = WaypointGraphGenerator(config=config)

# Generate graph with the custom configuration
graph = generator.build_graph_from_grid_map(
    image=occupancy_grid,
    occupancy_threshold=occupancy_threshold,
    safety_distance=safety_distance,
    resolution=resolution,
    x_offset=x_offset,
    y_offset=y_offset,
    rotation=rotation,
)
```

Note that all distance parameters are specified in meters (real-world units), not in pixels. The implementation internally converts these values to pixel units based on the provided resolution parameter.

### Manage Multiple Maps with Graph Manager

The `GraphManager` class provides a way to manage multiple maps and their associated waypoint graphs simultaneously. It handles graph generation, storage, visualization, and route finding across different maps in a thread-safe manner. Each map managed by the `GraphManager` should be identified with a unique ID. See [graph_manager_example.py](../examples/graph_manager_example.py) for a complete example of using the `GraphManager` class.

## REST API

The REST API provides a simple way to try out waypoint graph generation without writing Python code. To run the REST API,

```bash
# Select one of the below:
# 1. With the Python script, requires installation of the SWAGGER library.
python scripts/rest_api.py
# OR
# 2. With docker compose, installation of the library is optional.
cd docker
docker compose up rest-api
```

By default, the REST API service runs on `http://localhost:8000`. Visit `http://localhost:8000/v1/docs` for an interactive documentation page. If accessing from a different machine, replace `localhost` with the host name or IP address of the server running the REST API.

**Note:** The documentation page is generated by FastAPI and may become unresponsive when working with large graphs. This is a limitation of the browser-based UI rather than the API itself.

A brief explanation of available endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Checks the health status of the service. Returns status (RUNNING/WARNING) and any errors. |
| `/v1/graph/generate` | POST | Creates or updates a waypoint graph using a map file and required data. Accepts map files as uploads or from URI. |
| `/v1/graph` | GET | Retrieves the generated graph for a specific map ID. |
| `/v1/graph/nearest_nodes` | POST | Finds the nearest waypoint nodes to a list of points on a specified map. |
| `/v1/graph/visualize` | GET | Gets a visualization of the graph overlaid on the map as a PNG image. |
| `/v1/graph/route` | POST | Finds the shortest route between a start and goal point on the specified map. |

**Note**: Unlike the Python API, the graph building parameters cannot be adjusted when using the REST API.
