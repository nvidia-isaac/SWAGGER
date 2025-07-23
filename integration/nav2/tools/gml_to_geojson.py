#!/usr/bin/env python3
"""
GML to GeoJSON Converter

Converts Graph Modeling Language (GML) files to GeoJSON format for Nav2 Route Server.
The GML file is expected to contain nodes with world coordinates and edges connecting them.
"""

import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple


def parse_gml_file(gml_file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse a GML file and extract nodes and edges.
    
    Args:
        gml_file_path: Path to the GML file
        
    Returns:
        Tuple of (nodes, edges) lists
    """
    nodes = []
    edges = []
    
    with open(gml_file_path, 'r') as file:
        content = file.read()
    
    # Parse nodes
    node_pattern = r'node \[(.*?)\]'
    node_matches = re.findall(node_pattern, content, re.DOTALL)
    
    for node_match in node_matches:
        node_data = {}
        
        # Extract node attributes
        id_match = re.search(r'id (\d+)', node_match)
        if id_match:
            node_data['id'] = int(id_match.group(1))
        
        label_match = re.search(r'label "([^"]*)"', node_match)
        if label_match:
            node_data['label'] = label_match.group(1)
        
        # Extract world coordinates (x, y, z)
        world_matches = re.findall(r'world ([\d.-]+)', node_match)
        if len(world_matches) >= 2:
            node_data['x'] = float(world_matches[0])
            node_data['y'] = float(world_matches[1])
            if len(world_matches) >= 3:
                node_data['z'] = float(world_matches[2])
        
        # Extract pixel coordinates
        pixel_matches = re.findall(r'pixel (\d+)', node_match)
        if len(pixel_matches) >= 2:
            node_data['pixel_x'] = int(pixel_matches[0])
            node_data['pixel_y'] = int(pixel_matches[1])
        
        if 'id' in node_data and 'x' in node_data and 'y' in node_data:
            nodes.append(node_data)
    
    # Parse edges
    edge_pattern = r'edge \[(.*?)\]'
    edge_matches = re.findall(edge_pattern, content, re.DOTALL)
    
    for edge_match in edge_matches:
        edge_data = {}
        
        # Extract edge attributes
        source_match = re.search(r'source (\d+)', edge_match)
        if source_match:
            edge_data['source'] = int(source_match.group(1))
        
        target_match = re.search(r'target (\d+)', edge_match)
        if target_match:
            edge_data['target'] = int(target_match.group(1))
        
        weight_match = re.search(r'weight ([\d.-]+)', edge_match)
        if weight_match:
            edge_data['weight'] = float(weight_match.group(1))
        
        edge_type_match = re.search(r'edge_type "([^"]*)"', edge_match)
        if edge_type_match:
            edge_data['edge_type'] = edge_type_match.group(1)
        
        if 'source' in edge_data and 'target' in edge_data:
            edges.append(edge_data)
    
    return nodes, edges


def create_geojson(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """
    Convert nodes and edges to GeoJSON format.
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        
    Returns:
        GeoJSON dictionary
    """
    # Create a lookup for nodes by ID
    node_lookup = {node['id']: node for node in nodes}
    
    features = []
    feature_id = 0
    
    # Convert nodes to Point features
    for node in nodes:
        feature = {
            "type": "Feature",
            "properties": {
                "id": node['id'],
                "frame": "map"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [node['x'], node['y']]
            }
        }
        
        features.append(feature)
        feature_id += 1
    
    # Convert edges to LineString features
    for edge in edges:
        source_node = node_lookup.get(edge['source'])
        target_node = node_lookup.get(edge['target'])
        
        if source_node and target_node:
            # Forward direction edge
            forward_feature = {
                "type": "Feature",
                "properties": {
                    "id": feature_id,
                    "startid": edge['source'],
                    "endid": edge['target']
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[
                        [source_node['x'], source_node['y']],
                        [target_node['x'], target_node['y']]
                    ]]
                }
            }
            
            features.append(forward_feature)
            feature_id += 1
            
            # Reverse direction edge
            reverse_feature = {
                "type": "Feature",
                "properties": {
                    "id": feature_id,
                    "startid": edge['target'],
                    "endid": edge['source']
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[
                        [target_node['x'], target_node['y']],
                        [source_node['x'], source_node['y']]
                    ]]
                }
            }
            
            features.append(reverse_feature)
            feature_id += 1
    
    # Create the complete GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "name": "graph",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::3857"
            }
        },
        "date_generated": datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"),
        "features": features
    }
    
    return geojson


def main():
    """Main function to handle command line arguments and conversion."""
    parser = argparse.ArgumentParser(description='Convert GML file to GeoJSON format')
    parser.add_argument('input_file', help='Input GML file path')
    parser.add_argument('-o', '--output', help='Output GeoJSON file path (default: input_file.geojson)')
    
    args = parser.parse_args()
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        output_file = args.input_file.rsplit('.', 1)[0] + '.geojson'
    
    try:
        print(f"Parsing GML file: {args.input_file}")
        nodes, edges = parse_gml_file(args.input_file)
        print(f"Found {len(nodes)} nodes and {len(edges)} edges")
        
        print("Converting to GeoJSON...")
        geojson = create_geojson(nodes, edges)
        
        print(f"Writing GeoJSON to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=4)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
