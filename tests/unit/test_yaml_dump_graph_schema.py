import unittest
from pprint import pprint
import yaml  # For parsing and verifying the YAML output

from graph_nd.graphrag.graph_schema import GraphSchema, QueryPattern, RelationshipSchema, PropertySchema, NodeSchema


class TestGraphSchemaToYAML(unittest.TestCase):
    def test_to_yaml_serialization(self):
        # Create example nodes
        node = NodeSchema(
            description="A person node",
            id=PropertySchema(
                description="The person's unique ID",
                name="person_id",
                type="STRING"
            ),
            label="Person",
            properties=[
                PropertySchema(
                    description="The person's name",
                    name="name",
                    type="STRING"
                )
            ]
        )

        # Create example relationships
        relationship = RelationshipSchema(
            description="Knows relationship",
            type="KNOWS",
            queryPatterns=[
                QueryPattern(
                    description="A relationship from Person to Person",
                    startNode="Person",
                    endNode="Person"
                )
            ]
        )

        # Create the GraphSchema instance
        graph = GraphSchema(
            description="A sample graph schema",
            nodes=[node],
            relationships=[relationship]
        )

        # Serialize the graph to YAML
        graph_yaml = graph.query_model_to_yaml()

        # Print YAML for debugging purposes
        print("\nGenerated YAML Output:")
        print(graph_yaml)

        # Validate the YAML output
        try:
            parsed_yaml = yaml.safe_load(graph_yaml)  # Parse the YAML output back into a dictionary
        except yaml.YAMLError as e:
            self.fail(f"YAML serialization produced invalid YAML: {e}")

        # Assertions to ensure YAML structure matches expectations
        self.assertEqual(parsed_yaml["description"], "A sample graph schema")
        self.assertEqual(len(parsed_yaml["nodes"]), 1)
        self.assertEqual(parsed_yaml["nodes"][0]["label"], "Person")
        self.assertEqual(parsed_yaml["nodes"][0]["id"]["name"], "person_id")
        self.assertEqual(parsed_yaml["nodes"][0]["properties"][0]["name"], "name")
        self.assertEqual(parsed_yaml["nodes"][0]["properties"][0]["type"], "STRING")

        self.assertEqual(len(parsed_yaml["relationships"]), 1)
        self.assertEqual(parsed_yaml["relationships"][0]["type"], "KNOWS")
        self.assertEqual(parsed_yaml["relationships"][0]["queryPatterns"], ["(:Person)-[:KNOWS]->(:Person)"])
        self.assertEqual(parsed_yaml["relationships"][0]["description"], "Knows relationship")
