# GML to GeoJSON Converter

A Python utility for converting Graph Modeling Language (GML) files to GeoJSON format for use with Nav2 Route Server.

## Overview

The `gml_to_geojson.py` script parses GML files containing graph data with nodes and edges, converting them to GeoJSON format suitable for navigation and routing applications.

## Features

- Parses GML nodes with world coordinates (x, y, z) and pixel coordinates
- Extracts edges with source/target relationships
- Generates bidirectional LineString features for each edge
- Outputs valid GeoJSON with EPSG:3857 coordinate reference system
- Supports pretty-printed JSON output

## Usage

```bash
# Basic conversion
python3 gml_to_geojson.py input.gml

# Specify output file
python3 gml_to_geojson.py input.gml -o output.geojson
```

## Arguments

- `input_file`: Path to the input GML file (required)
- `-o, --output`: Output GeoJSON file path (default: input_file.geojson)

## GML Format Requirements

The script expects GML files with the following structure:

**Nodes:**
```
node [
    id 1
    label "node_name"
    world 123.45 67.89 0.0
    pixel 100 200
]
```

**Edges:**
```
edge [
    source 1
    target 2
    weight 1.5
    edge_type "normal"
]
```

## Output Format

The generated GeoJSON contains:
- Point features for each node with world coordinates
- Bidirectional MultiLineString features for each edge
- EPSG:3857 coordinate reference system
- Generation timestamp

## Dependencies

- Python 3.x
- Standard library modules: `re`, `json`, `argparse`, `datetime`, `typing`
