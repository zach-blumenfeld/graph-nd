import unittest
import json
from pathlib import Path

from graph_nd.graphrag.graph_schema import GraphSchema


class TestGraphSchemaSubset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load the movie schema JSON file once for all tests.
        """
        with open(Path("data/movie-schema-animal.json"), "r") as file:
            cls.graph_data = json.load(file)
        cls.graph_schema = GraphSchema(**cls.graph_data)

    def test_subset_with_nodes_only(self):
        """
        Test subset behavior when filtering with just nodes.
        """
        subset = self.graph_schema.subset(nodes=["Person", "Movie"])
        self.assertEqual(len(subset.nodes), 2)
        self.assertTrue(any(node.label == "Person" for node in subset.nodes))
        self.assertTrue(any(node.label == "Movie" for node in subset.nodes))
        self.assertEqual(len(subset.relationships), 0, "Expected no relationships in the subset")

    def test_subset_with_patterns_only(self):
        """
        Test subset behavior when filtering with just patterns.
        """
        subset = self.graph_schema.subset(patterns=[("Animal", "ACTED_IN", "Movie")])
        self.assertEqual(len(subset.relationships), 1)
        # Extract the first relationship and assert the query pattern
        rel = subset.relationships[0]
        self.assertEqual(rel.type, "ACTED_IN")
        self.assertEqual(len(rel.queryPatterns), 1, "Expected exactly one query pattern in the relationship")
        pattern = rel.queryPatterns[0]
        self.assertEqual(pattern.startNode, "Animal", "Expected startNode to be 'Animal'")
        self.assertEqual(pattern.endNode, "Movie", "Expected endNode to be 'Movie'")

    def test_subset_with_relationships_only(self):
        """
        Test subset behavior when filtering with just relationships.
        """
        subset = self.graph_schema.subset(relationships=["ACTED_IN", "DIRECTED"])

        # Assert both relationships are present
        acted_in = next((rel for rel in subset.relationships if rel.type == "ACTED_IN"), None)
        directed = next((rel for rel in subset.relationships if rel.type == "DIRECTED"), None)

        self.assertIsNotNone(acted_in, "Expected the ACTED_IN relationship to be in the subset")
        self.assertIsNotNone(directed, "Expected the DIRECTED relationship to be in the subset")

        # Validate ACTED_IN query patterns
        self.assertEqual(len(acted_in.queryPatterns), 2, "Expected exactly two query patterns in ACTED_IN")

        # Check both query patterns in ACTED_IN
        patterns_acted_in = {(pattern.startNode, pattern.endNode) for pattern in acted_in.queryPatterns}
        self.assertIn(("Person", "Movie"), patterns_acted_in,
                      "Expected a query pattern with startNode 'Person' and endNode 'Movie' in ACTED_IN")
        self.assertIn(("Animal", "Movie"), patterns_acted_in,
                      "Expected a query pattern with startNode 'Animal' and endNode 'Movie' in ACTED_IN")

        # Validate DIRECTED query patterns
        self.assertEqual(len(directed.queryPatterns), 1, "Expected exactly one query pattern in DIRECTED")
        patterns_directed = {(pattern.startNode, pattern.endNode) for pattern in directed.queryPatterns}
        self.assertIn(("Person", "Movie"), patterns_directed,
                      "Expected a query pattern with startNode 'Person' and endNode 'Movie' in DIRECTED")

    def test_subset_with_combination(self):
        """
        Test subset behavior with a combination of nodes, patterns, and relationships.
        """
        subset = self.graph_schema.subset(
            nodes=["Award"],
            patterns=[("Person", "ACTED_IN", "Movie")],
            relationships=["ACTED_IN"]
        )
        # the acted_in relationship implies Animal node as well so we need 4 nodes total
        self.assertEqual(len(subset.nodes), 4)
        # we only have one relationship
        self.assertEqual(len(subset.relationships), 1)
        # assert we have all query patterns
        acted_in = next((rel for rel in subset.relationships if rel.type == "ACTED_IN"), None)
        self.assertIsNotNone(acted_in, "Expected the ACTED_IN relationship to be in the subset")
        # Check both query patterns in ACTED_IN
        patterns_acted_in = {(pattern.startNode, pattern.endNode) for pattern in acted_in.queryPatterns}
        self.assertIn(("Person", "Movie"), patterns_acted_in,
                      "Expected a query pattern with startNode 'Person' and endNode 'Movie' in ACTED_IN")
        self.assertIn(("Animal", "Movie"), patterns_acted_in,
                      "Expected a query pattern with startNode 'Animal' and endNode 'Movie' in ACTED_IN")

    def test_subset_with_invalid_nodes(self):
        """
        Verify that an error is raised when requesting invalid nodes.
        """
        node_label = "InvalidNode"
        with self.assertRaises(ValueError) as cm:
            self.graph_schema.subset(nodes=node_label)
        self.assertEqual(str(cm.exception), f"No NodeSchema found with the label '{node_label}'")

    def test_subset_with_invalid_patterns(self):
        """
        Verify that an error is raised for invalid patterns (startNode/endNode combination).
        """
        start_node_label = "Animal"
        rel_type = "DIRECTED"
        end_node_label = "Movie"
        with self.assertRaises(ValueError) as cm:
            self.graph_schema.subset(patterns=[(start_node_label, rel_type, end_node_label)])
        self.assertEqual(
            str(cm.exception),
            f"No RelationshipSchema found with type '{rel_type}' and query pattern "
            f"'{start_node_label}-[{rel_type}]->{end_node_label}'"
        )

    def test_subset_with_invalid_relationships(self):
        """
        Verify that an error is raised when requesting invalid relationships.
        """
        rel_type = "INVALID_REL"
        with self.assertRaises(ValueError) as cm:
            self.graph_schema.subset(relationships=[rel_type])
        self.assertEqual(str(cm.exception), f"No RelationshipSchema found with type '{rel_type}'")


if __name__ == '__main__':
    unittest.main()
