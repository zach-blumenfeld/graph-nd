import json
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from graph_nd.graphrag.graph_schema import GraphSchema  # Replace with the real path for GraphSchema
from graph_nd.graphrag.graph_data import RelationshipData  # Import your RelationshipData class

# Filepath for `movie-schema.json`
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data/movie-schema.json')


class TestRelationshipDataMerge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the GraphSchema from the JSON file once for all test cases"""
        with open(SCHEMA_FILE_PATH, "r") as f:
            data = json.load(f)
            cls.graph_schema = GraphSchema(**data)  # Parse JSON into GraphSchema

    def setUp(self):
        """Set up test instance variables"""
        self.db_client = MagicMock()  # Mock database client
        self.rel_schema = self.graph_schema.get_relationship_schema("ACTED_IN", "Person", "Movie")
        self.assertIsNotNone(self.rel_schema, "Relationship schema for ACTED_IN not found!")

        # Initialize RelationshipData with actual schema
        self.rel_data = RelationshipData(
            rel_schema=self.rel_schema,
            start_node_schema=self.graph_schema.get_node_schema_by_label("Person"),
            end_node_schema=self.graph_schema.get_node_schema_by_label("Movie"),
            records=[{"start_node_id": "person_1", "end_node_id": "movie_1", "role": "role_1"}]
        )

    @patch("graph_nd.graphrag.graph_data.validate_and_create_source_node")  # Patch standalone function
    def test_merge_with_source_id(self, mock_validate):
        """Test merging with source metadata"""
        source_metadata = {"id": "source123"}
        self.rel_data.merge(self.db_client, source_metadata=source_metadata)

        # Assert if validate_and_create_source_node was called
        mock_validate.assert_called_once_with({'id': 'source123', 'sourceType': 'RELATIONSHIP_LIST', 'transformType': 'UNKNOWN', 'loadType': 'MERGE_RELATIONSHIPS', 'name': 'relationship-merge'},self.db_client)

    @patch("graph_nd.graphrag.graph_data.validate_and_create_source_node")  # Patch standalone function
    def test_merge_without_source_id(self, mock_validate):
        """Test merging without source metadata"""
        self.rel_data.merge(self.db_client, source_metadata=False)

        # Assert validate_and_create_source_node was NOT called
        mock_validate.assert_not_called()
