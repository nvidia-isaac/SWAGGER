# Evaluation
To evaluate the generated graph, run the script in the repository root:
```bash
python scripts/evaluate_graph.py \
    --graph_path <path_to_graph.gml> \
    --map_path <path_to_occupancy_grid_map.png> \
    --resolution <resolution> \
    --safety-distance <safety-distance> \
    --occupancy-threshold <occupancy-threshold> \
    --output_dir <output_dir>
```

## Coverage Metrics
These metrics evaluate how well the graph covers the free space in the environment.

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Free Space Coverage** | The percentage of free space that has at least one node within it.          |
| **Average Distance to Node** | The average distance from free cells to the nearest node in the graph, in pixels. Divide it by the resolution to get meters. |
| **Max Distance to Node** | The maximum distance from any free cell to the nearest node, in pixels. Divide it by the resolution to get meters. |
| **Coverage Efficiency** | The ratio of free space coverage to the number of nodes in the graph.       |

### Graph Structure Metrics
These metrics provide insights into the structural properties of the graph.

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Number of Nodes**     | The total number of nodes in the graph.                                     |
| **Number of Edges**     | The total number of edges connecting the nodes.                             |
| **Average Node Degree** | The average number of connections (edges) per node.                         |
| **Average Edge Length** | The average length of the edges in pixels.                                  |
| **Graph Diameter**      | The longest shortest path between any two nodes in the graph.               |
| **Average Clustering**  | The average clustering coefficient, indicating the degree to which nodes tend to cluster together. |

### Path Planning Metrics
These metrics assess the quality of paths that can be planned using the graph.

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Average Path Length** | The average length of the shortest paths between random node pairs.         |
| **Average Path Smoothness** | A measure of the smoothness of paths, based on the angles between consecutive edges. |
| **Path Success Rate**   | The percentage of valid paths found between random node pairs.              |
| **Average Path Clearance** | The average distance to obstacles along the paths.                       |

### Validation Metrics
These metrics validate the correctness and feasibility of the graph.

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Collision Free**      | Indicates whether all edges are free of collisions with obstacles.          |
| **Node Validity**       | Checks if all nodes are in valid positions within the free space.           |
| **Connectivity**        | Determines if the graph is fully connected.                                 |
| **Number of Subgraphs** | The number of disconnected subgraphs within the graph.                      |
| **Edge Length Validity**| Ensures that edge lengths are within reasonable bounds.                     |
