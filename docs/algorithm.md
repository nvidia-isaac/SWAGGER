# Algorithm

The following map is used to illustrate the algorithm:

   <picture>
      <source srcset="images/carter_warehouse_navigation.png">
      <img src="images/carter_warehouse_navigation.png" width="400" alt="Original Map" style="display: block; margin: 0 auto;" class="center">
   </picture>

1. **Safety Buffer Addition**:
   - Compute the distance transform of the free space to determine the shortest distance to the nearest obstacle for each free pixel.
   - Mark pixels within the safety buffer as obstacles.

   <picture>
      <source srcset="images/inflated_map.png">
      <img src="images/inflated_map.png" width="400" alt="Inflated Map" style="display: block; margin: 0 auto;" class="center">
   </picture>


2. **Skeleton Graph Construction**:
   - Build the graph based on the skeleton of the map.
   - Identify branches in the skeleton and add nodes from one end to the other, connecting consecutive nodes.

   <picture>
      <source srcset="images/skeleton_only.png">
      <img src="images/skeleton_only.png" width="400" alt="Skeleton Map" style="display: block; margin: 0 auto;" class="center">
   </picture>


3. **Boundary Node Creation**:
   - Increase the safety buffer around obstacles.
   - Identify contours of the obstacles and add nodes along these contours to improve node distribution near obstacles.

   <picture>
      <source srcset="images/skeleton_and_boundary.png">
      <img src="images/skeleton_and_boundary.png" width="400" alt="Skeleton and Boundary Map" style="display: block; margin: 0 auto;" class="center">
   </picture>


4. **Node Placement in Free Space**:
   - Compute the distance transform of the free space to existing nodes.
   - Place nodes at local maxima that are sufficiently distant from existing nodes.
   - Repeat until no more suitable local maxima are found.

   <picture>
      <source srcset="images/free_space_sampling.png">
      <img src="images/free_space_sampling.png" width="400" alt="Free Space Sampling" style="display: block; margin: 0 auto;" class="center">
   </picture>


5. **Triangulation and Edge Addition**:
   - Perform Delaunay triangulation on the nodes.
   - Add edges between nodes connected by the triangulation.

   <picture>
      <source srcset="images/shortcut.png">
      <img src="images/shortcut.png" width="400" alt="Shortcutting" style="display: block; margin: 0 auto;" class="center">
   </picture>


6. **Graph Pruning**:
   - Remove isolated nodes, small subgraphs, and node clusters to refine the graph.

   <picture>
      <source srcset="images/prune.png">
      <img src="images/prune.png" width="400" alt="Pruning" style="display: block; margin: 0 auto;" class="center">
   </picture>
