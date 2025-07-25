# SWAGGER: Sparse WAypoint Graph Generation for Efficient Routing

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/ubuntu-22.04-red)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

A Python package that generates sparse waypoint graphs from occupancy grid maps for route planning.

<img src="docs/images/generation_in_action.gif" alt="Graph Generation in Action" height="300"/>

See [algorithm overview](docs/algorithm.md) for an introduction of the method.


## Table of Contents
- [Algorithm Overview](docs/algorithm.md)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Set up the Environment](#set-up-the-environment)
  - [Install Package](#install-package)
- [Usage](#usage)
  - [Data Prepration](#data-preparation)
  - [Command Line Interface](#command-line-interface)
  - [Evaluation](#evaluation)
  - [REST API](#rest-api-service)
  - [Tutorial](#tutorial)
  - [Integration examples](integration/README.md)
- [Development](#development)
  - [Testing](#testing)
  - [Linting and Formatting](#linting-and-formatting)


## System Requirements
* Python 3.10 or newer
* [CUDA 12.5](https://developer.nvidia.com/cuda-12-5-0-download-archive) or newer (including NVIDIA CUDA toolkit)

## Installation

1. Clone the repository.

    ```bash
    git clone git@github.com:nvidia-isaac/SWAGGER.git
    cd SWAGGER
    git lfs pull
    ```

2. Install required packages.
```bash
sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

3. Start a virtual environment.
```bash
python -m venv swagger-venv
source swagger-venv/bin/activate
```

4. Install the SWAGGER library.
```bash
pip install -e .
```

## Usage

### Data Preparation

The following data is required for waypoint graph generation:

* **Occupancy Grid Map**:
    * **Image Format**: A grayscale image where each pixel represents the occupancy probability in a 2D world:
        * 0 (black) = completely occupied
        * 255 (white) = completely free
    * **Occupancy Threshold**: A value that determines which pixels are considered free space. Pixels with values above this threshold will be used for graph generation.

* **Robot Parameters**:
    * **Safety Distance**: The minimum distance (in meters) that graph nodes and edges must maintain from obstacles to ensure safe robot navigation.

* **Coordinate Transform**: Parameters to convert pixel coordinates to real-world coordinates:
    * **Resolution**: The size of each pixel in meters (meters/pixel)
    * **X Offset**: The X coordinate of the bottom-left pixel in the world frame in meters.
    * **Y Offset**: The Y coordinate of the bottom-left pixel in the world frame in meters.
    * **Rotation**: Rotation angle from image frame to world frame in radians.

#### Coordinate Transform

In the image frame:
- Origin (0,0) is at the top-left corner
- X-axis points down (rows)
- Y-axis points right (columns)

When converting from image to world coordinates:
1. Image coordinates are scaled by the resolution to convert from pixels to meters
2. Rotation is applied around the origin using the provided rotation angle
3. X and Y offsets are added to translate the coordinates to the final world position

![coordinate_transform](docs/images/coordinate_transform.png)

### Command Line Interface
Generate a waypoint graph from an occupancy grid map:
```bash
python scripts/generate_graph.py \
    --map-path <path_to_map.png> \
    --resolution <meters_per_pixel> \
    --safety-distance <meters> \
    --output-dir <output_directory>
```

Default parameters:
- Map: `maps/carter_warehouse_navigation.png`
- Resolution: 0.05 meters/pixel
- Safety distance: 0.3 meters

Outputs:
- `<output_dir>/waypoint_graph.png`: Visualization of the generated graph
- `<output_dir>/graph.gml`: Graph data in GML format

### Evaluation

To evaluate the graph, see the tutorial on [evaluation](docs/evaluation.md).

### REST API Service

Start the service locally:
```bash
python scripts/rest_api.py
```

The service will be available at `http://localhost:8000`. Visit `http://localhost:8000/v1/docs` in your browser to view the interactive API documentation. If accessing from a different machine, replace `localhost` with the host name or the IP address of the server running the REST API. For more information about the REST API, check out the [tutorial](docs/tutorial.md#rest-api).

The REST API is also available as a docker compose service.

```bash
cd docker
docker compose build swagger
docker compose up rest-api
```

We provide a sample script, `scripts/test_api_client.py`, to demonstrate the usage the REST API service. With the REST API service running in a separate terminal, run the following command in your virtual environment:
```bash
python scripts/test_api_client.py --map_path maps/carter_warehouse_navigation.png
```


### Tutorial

This [tutorial](docs/tutorial.md) provides a comprehensive guide on how to use the SWAGGER library, including graph building, visualization, route finding, parameter tuning, and using both the Python API and REST API interfaces.


## Development

### Testing
```bash
python -m unittest
```

### Linting and Formatting
This project uses pre-commit hooks for linting and formatting:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Run manually
```
