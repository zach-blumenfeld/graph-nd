import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from graph_schema import GraphSchema  # Replace with the real path for GraphSchema
from graph_data import NodeData  # Import your NodeData class from the correct module

# Filepath for `movie-schema.json`
SCHEMA_FILE_PATH = Path("data/movie-schema.json")


class TestNodeDataMerge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the GraphSchema from the JSON file once for all test cases"""
        with open(SCHEMA_FILE_PATH, "r") as f:
            data = json.load(f)
            cls.graph_schema = GraphSchema(**data)  # Parse JSON into GraphSchema

    def setUp(self):
        """Set up test instance variables"""
        self.db_client = MagicMock()  # Mock database client
        self.person_schema = self.graph_schema.get_node_schema_by_label("Person")
        self.assertIsNotNone(self.person_schema, "Person node schema not found!")

        # Initialize NodeData with actual schema
        self.node_data = NodeData(
            node_schema=self.person_schema,
            records=[{"name": "Alice", "age": 30}]
        )

    @patch("graph_data.validate_and_create_source_node")  # Patch the standalone function
    def test_merge_no_metadata(self, mock_validate):
        """Test merging without metadata"""
        self.node_data.merge(self.db_client, source_metadata=False)

        # Assert validate_and_create_source_node was NOT called
        mock_validate.assert_not_called()

    @patch("graph_data.validate_and_create_source_node")  # Patch the standalone function
    def test_merge_custom_metadata(self, mock_validate):
        """Test merging with custom metadata"""
        source_metadata = {"id": "source123", "name": "custom"}
        self.node_data.merge(self.db_client, source_metadata=source_metadata)

        # Assert validate_and_create_source_node was called with the correct arguments
        mock_validate.assert_called_once_with({'id': 'source123', 'name': 'custom', 'sourceType': 'NODE_LIST', 'transformType': 'UNKNOWN', 'loadType': 'MERGE_NODES'},
                                              self.db_client)

