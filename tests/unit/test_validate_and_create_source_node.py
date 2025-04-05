import unittest
import uuid
from unittest.mock import MagicMock

from neo4j import RoutingControl

from graphrag.graph_data import validate_and_create_source_node


class TestValidateAndCreateSourceNode(unittest.TestCase):

    def setUp(self):
        self.db_client = MagicMock()

    def test_with_valid_metadata(self):
        source_metadata = {"id": "test_id", "name": "test_name"}
        result_id = validate_and_create_source_node(source_metadata, self.db_client)

        self.assertEqual(result_id, "test_id")
        self.db_client.execute_query.assert_any_call(
            'CREATE INDEX range___source___id IF NOT EXISTS FOR (n:__Source__) ON n.id',
            routing_=RoutingControl.WRITE
        )
        self.db_client.execute_query.assert_any_call(
            """MERGE(n:__Source__ {id: $rec.id, name: $rec.name}) SET n.createdAt = datetime.realtime()""",
            routing_=RoutingControl.WRITE,
            rec=source_metadata
        )

    def test_with_partial_metadata(self):
        source_metadata = {"id": ""}
        result_id = validate_and_create_source_node(source_metadata, self.db_client)

        # Assert that `id` is replaced with a valid UUID
        self.assertTrue(uuid.UUID(result_id))
        self.assertNotEqual(result_id, "")  # Ensure `id` is not empty