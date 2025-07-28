#!/usr/bin/env python3
"""
Tests for GML to GeoJSON Converter

Tests the functionality of the gml_to_geojson.py module including
GML parsing, GeoJSON conversion, and command line interface.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import mock_open, patch

# Add the parent directory to the path to import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gml_to_geojson import create_geojson, main, parse_gml_file  # noqa: E402


class TestGMLParser(unittest.TestCase):
    """Test GML file parsing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_gml = """graph [
node [
id 1
label "node1"
world 10.5
world 20.3
world 0.0
pixel 100
pixel 200
]
node [
id 2
label "node2"
world -5.2
world 15.7
pixel 150
pixel 250
]
node [
id 3
label "node3"
world 0.0
world 0.0
world 5.5
]
edge [
source 1
target 2
weight 3.14
edge_type "normal"
]
edge [
source 2
target 3
weight 2.5
]
]"""

        self.invalid_gml = """invalid content
        no proper structure
        """

        self.partial_gml = """graph [
node [
id 1
world 10.5
world 20.3
]
node [
id 2
]
edge [
source 1
]
]"""

    def test_parse_valid_gml_file(self):
        """Test parsing a valid GML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gml", delete=False) as f:
            f.write(self.sample_gml)
            f.flush()

            try:
                nodes, edges = parse_gml_file(f.name)

                # Test nodes
                self.assertEqual(len(nodes), 3)

                # Test first node
                node1 = next(n for n in nodes if n["id"] == 1)
                self.assertEqual(node1["label"], "node1")
                self.assertEqual(node1["x"], 10.5)
                self.assertEqual(node1["y"], 20.3)
                self.assertEqual(node1["z"], 0.0)
                self.assertEqual(node1["pixel_x"], 100)
                self.assertEqual(node1["pixel_y"], 200)

                # Test second node (no z coordinate)
                node2 = next(n for n in nodes if n["id"] == 2)
                self.assertEqual(node2["label"], "node2")
                self.assertEqual(node2["x"], -5.2)
                self.assertEqual(node2["y"], 15.7)
                self.assertNotIn("z", node2)

                # Test third node (no pixel coordinates)
                node3 = next(n for n in nodes if n["id"] == 3)
                self.assertEqual(node3["x"], 0.0)
                self.assertEqual(node3["y"], 0.0)
                self.assertEqual(node3["z"], 5.5)
                self.assertNotIn("pixel_x", node3)

                # Test edges
                self.assertEqual(len(edges), 2)

                # Test first edge
                edge1 = next(e for e in edges if e["source"] == 1 and e["target"] == 2)
                self.assertEqual(edge1["weight"], 3.14)
                self.assertEqual(edge1["edge_type"], "normal")

                # Test second edge
                edge2 = next(e for e in edges if e["source"] == 2 and e["target"] == 3)
                self.assertEqual(edge2["weight"], 2.5)
                self.assertNotIn("edge_type", edge2)

            finally:
                os.unlink(f.name)

    def test_parse_partial_gml_file(self):
        """Test parsing GML file with missing or incomplete data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gml", delete=False) as f:
            f.write(self.partial_gml)
            f.flush()

            try:
                nodes, edges = parse_gml_file(f.name)

                # Only nodes with id, x, and y should be included
                self.assertEqual(len(nodes), 1)
                self.assertEqual(nodes[0]["id"], 1)

                # Edges without both source and target should be excluded
                self.assertEqual(len(edges), 0)

            finally:
                os.unlink(f.name)

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                nodes, edges = parse_gml_file(f.name)
                self.assertEqual(len(nodes), 0)
                self.assertEqual(len(edges), 0)
            finally:
                os.unlink(f.name)

    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            parse_gml_file("/nonexistent/file.gml")


class TestGeoJSONCreation(unittest.TestCase):
    """Test GeoJSON creation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.nodes = [
            {"id": 1, "x": 10.5, "y": 20.3, "label": "node1"},
            {"id": 2, "x": -5.2, "y": 15.7, "label": "node2"},
            {"id": 3, "x": 0.0, "y": 0.0, "label": "node3"},
        ]

        self.edges = [{"source": 1, "target": 2, "weight": 3.14}, {"source": 2, "target": 3, "weight": 2.5}]

    def test_create_geojson_structure(self):
        """Test that GeoJSON has correct structure."""
        geojson = create_geojson(self.nodes, self.edges)

        # Test basic structure
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(geojson["name"], "graph")
        self.assertIn("crs", geojson)
        self.assertIn("date_generated", geojson)
        self.assertIn("features", geojson)

        # Test CRS
        self.assertEqual(geojson["crs"]["type"], "name")
        self.assertEqual(geojson["crs"]["properties"]["name"], "urn:ogc:def:crs:EPSG::3857")

    def test_create_geojson_nodes(self):
        """Test that nodes are correctly converted to Point features."""
        geojson = create_geojson(self.nodes, [])
        features = geojson["features"]

        # Should have 3 point features
        point_features = [f for f in features if f["geometry"]["type"] == "Point"]
        self.assertEqual(len(point_features), 3)

        # Test first node
        node1_feature = next(f for f in point_features if f["properties"]["id"] == 1)
        self.assertEqual(node1_feature["type"], "Feature")
        self.assertEqual(node1_feature["properties"]["frame"], "map")
        self.assertEqual(node1_feature["geometry"]["coordinates"], [10.5, 20.3])

    def test_create_geojson_edges(self):
        """Test that edges are correctly converted to LineString features."""
        geojson = create_geojson(self.nodes, self.edges)
        features = geojson["features"]

        # Should have line features (2 edges * 2 directions = 4 line features)
        line_features = [f for f in features if f["geometry"]["type"] == "MultiLineString"]
        self.assertEqual(len(line_features), 4)

        # Test forward edge from node 1 to node 2
        forward_edge = next(
            f for f in line_features if f["properties"]["startid"] == 1 and f["properties"]["endid"] == 2
        )
        expected_coords = [[10.5, 20.3], [-5.2, 15.7]]
        self.assertEqual(forward_edge["geometry"]["coordinates"][0], expected_coords)

        # Test reverse edge from node 2 to node 1
        reverse_edge = next(
            f for f in line_features if f["properties"]["startid"] == 2 and f["properties"]["endid"] == 1
        )
        expected_coords = [[-5.2, 15.7], [10.5, 20.3]]
        self.assertEqual(reverse_edge["geometry"]["coordinates"][0], expected_coords)

    def test_create_geojson_missing_nodes(self):
        """Test handling of edges referencing non-existent nodes."""
        edges_with_missing = [
            {"source": 1, "target": 999},  # target doesn't exist
            {"source": 2, "target": 3},  # valid edge
        ]

        geojson = create_geojson(self.nodes, edges_with_missing)
        line_features = [f for f in geojson["features"] if f["geometry"]["type"] == "MultiLineString"]

        # Should only have 2 line features (1 valid edge * 2 directions)
        self.assertEqual(len(line_features), 2)

    def test_create_geojson_empty_input(self):
        """Test creating GeoJSON from empty nodes and edges."""
        geojson = create_geojson([], [])

        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 0)

    def test_date_generation(self):
        """Test that date is properly generated."""
        geojson = create_geojson([], [])
        # Just verify that a date string is generated
        self.assertIn("date_generated", geojson)
        self.assertIsInstance(geojson["date_generated"], str)
        self.assertGreater(len(geojson["date_generated"]), 0)


class TestMainFunction(unittest.TestCase):
    """Test the main function and command line interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_gml = """graph [
node [
id 1
label "test"
world 0.0
world 0.0
]
edge [
source 1
target 1
]
]"""

    @patch("sys.argv", ["gml_to_geojson.py", "test.gml"])
    @patch("builtins.open", new_callable=mock_open)
    @patch("gml_to_geojson.parse_gml_file")
    @patch("json.dump")
    def test_main_default_output(self, mock_json_dump, mock_parse, mock_file):
        """Test main function with default output filename."""
        mock_parse.return_value = ([], [])

        result = main()

        self.assertEqual(result, 0)
        mock_parse.assert_called_once_with("test.gml")
        # Check that output file was opened with .geojson extension
        mock_file.assert_called_with("test.geojson", "w")

    @patch("sys.argv", ["gml_to_geojson.py", "input.gml", "-o", "custom.geojson"])
    @patch("builtins.open", new_callable=mock_open)
    @patch("gml_to_geojson.parse_gml_file")
    @patch("json.dump")
    def test_main_custom_output(self, mock_json_dump, mock_parse, mock_file):
        """Test main function with custom output filename."""
        mock_parse.return_value = ([], [])

        result = main()

        self.assertEqual(result, 0)
        mock_parse.assert_called_once_with("input.gml")
        mock_file.assert_called_with("custom.geojson", "w")

    @patch("sys.argv", ["gml_to_geojson.py", "test.gml"])
    @patch("gml_to_geojson.parse_gml_file")
    def test_main_error_handling(self, mock_parse):
        """Test main function error handling."""
        mock_parse.side_effect = Exception("Test error")

        result = main()

        self.assertEqual(result, 1)

    def test_integration_full_workflow(self):
        """Integration test of the complete workflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gml", delete=False) as input_file:
            input_file.write(self.sample_gml)
            input_file.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as output_file:
                output_file.close()  # Close so main can write to it

                try:
                    # Test the complete workflow
                    with patch("sys.argv", ["gml_to_geojson.py", input_file.name, "-o", output_file.name]):
                        result = main()

                    self.assertEqual(result, 0)

                    # Verify output file was created and contains valid GeoJSON
                    with open(output_file.name) as f:
                        geojson_content = json.load(f)

                    self.assertEqual(geojson_content["type"], "FeatureCollection")
                    self.assertIn("features", geojson_content)

                finally:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)


if __name__ == "__main__":
    unittest.main()
