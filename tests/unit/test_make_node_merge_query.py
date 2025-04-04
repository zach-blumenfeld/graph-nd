import json
from unittest import TestCase
from pathlib import Path
from graph_schema import GraphSchema, NodeSchema
from graph_data import NodeData  

# Filepath for `movie-schema.json`
SCHEMA_FILE_PATH = Path("data/movie-schema.json")


class TestMakeNodeMergeQuery(TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the GraphSchema from the JSON file once for all test cases"""
        with open(SCHEMA_FILE_PATH, "r") as f:
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
