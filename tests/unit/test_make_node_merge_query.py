import json
import os
from unittest import TestCase
from pathlib import Path
from graph_nd.graphrag.graph_schema import GraphSchema, NodeSchema
from graph_nd.graphrag.graph_data import NodeData


class TestMakeNodeMergeQuery(TestCase):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def setUpClass(cls):
        """Load the GraphSchema from the JSON file once for all test cases"""
        with open(os.path.join(cls.DATA_DIR, "movie-schema.json"), "r") as f:
            data = json.load(f)
            cls.graph_schema = GraphSchema(**data)  # Parse JSON into GraphSchema

    def test_without_source_id(self):
        """Test merge query generation without source_id"""
        # Fetch the actual NodeSchema for Person
        person_schema = self.graph_schema.get_node_schema_by_label("Person")
        self.assertIsNotNone(person_schema, "Person node schema not found!")

        # Create NodeData using the actual schema
        node_data = NodeData(node_schema=person_schema, records=[{"name": "Alice"}])

        # Call the method to test
        result = node_data.make_node_merge_query()

        # Assertions to verify the generated query
        self.assertIn("SET n.name = rec.name", result)
        self.assertNotIn("SET n.__source_id", result)

    def test_with_source_id(self):
        """Test merge query generation with source_id"""
        # Fetch the actual NodeSchema for Person
        person_schema = self.graph_schema.get_node_schema_by_label("Person")
        self.assertIsNotNone(person_schema, "Person node schema not found!")

        # Create NodeData using the actual schema
        node_data = NodeData(node_schema=person_schema, records=[{"name": "Alice"}])

        # Call the method to test with a source ID
        result = node_data.make_node_merge_query(source_id="source123")

        # Assertions to verify the generated query
        self.assertIn("SET n.name = rec.name", result)
        self.assertIn(
            'SET n.__source_id = coalesce(n.__source_id, [])  + ["source123"]', result
        )
