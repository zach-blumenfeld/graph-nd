import unittest

from graph_nd.graphrag.graph_data import make_source_set_clause


class TestMakeSourceSetClause(unittest.TestCase):

    def test_default_clause(self):
        result = make_source_set_clause("source_id").strip()
        expected = 'SET n.__source_id = coalesce(n.__source_id, [])  + ["source_id"]'
        self.assertEqual(result, expected)

    def test_custom_element_clause(self):
        result = make_source_set_clause("source_id", element_name="r").strip()
        expected = 'SET r.__source_id = coalesce(r.__source_id, [])  + ["source_id"]'
        self.assertEqual(result, expected)
