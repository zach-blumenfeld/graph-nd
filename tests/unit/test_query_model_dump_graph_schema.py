import unittest
from pprint import pprint  # For nicely formatted printing

from GraphSchema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, GraphSchema


class TestGraphSchemaQueryModelDump(unittest.TestCase):
    def test_query_model_dump_graph_schema(self):
        # Create example nodes
        node_1 = NodeSchema(
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
                ),
                PropertySchema(
                    description="The person's age",
                    name="age",
                    type="INTEGER"
                )
            ]
        )

        node_2 = NodeSchema(
            description="A manager node",
            id=PropertySchema(
                description="The manager's unique ID",
                name="manager_id",
                type="STRING"
            ),
            label="Manager",
            properties=[
                PropertySchema(
                    description="The manager's department",
                    name="department",
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
                ),
                QueryPattern(
                    description="A relationship from Person to Manager",
                    startNode="Person",
                    endNode="Manager"
                )
            ],
            properties=[
                PropertySchema(
                    description="Year of the relationship",
                    name="since",
                    type="INTEGER"
                )
            ]
        )

        # Create the GraphSchema instance
        graph = GraphSchema(
            description="A sample graph schema",
            nodes=[node_1, node_2],
            relationships=[relationship]
        )

        # Print the full serialized graph to the console
        print("\nFull Serialized GraphSchema:")
        pprint(graph.query_model_dump())  # Print the entire graph to visually inspect the output

        # Get the serialized output using query_model_dump
        serialized_graph = graph.query_model_dump()

        # Assert nodes are correctly serialized
        self.assertEqual(len(serialized_graph["nodes"]), 2)
        self.assertEqual(serialized_graph["nodes"][0]["label"], "Person")
        self.assertEqual(serialized_graph["nodes"][1]["label"], "Manager")

        # Assert relationships are correctly serialized
        self.assertEqual(len(serialized_graph["relationships"]), 1)
        serialized_relationship = serialized_graph["relationships"][0]
        self.assertEqual(serialized_relationship["type"], "KNOWS")

        # Expected relationship patterns (from query_model_dump in RelationshipSchema)
        expected_query_patterns = [
            "(:Person)-[:KNOWS]->(:Person)",
            "(:Person)-[:KNOWS]->(:Manager)"
        ]
        self.assertEqual(serialized_relationship["queryPatterns"], expected_query_patterns)

        # Assert properties of the relationship
        self.assertEqual(len(serialized_relationship["properties"]), 1)
        self.assertEqual(serialized_relationship["properties"][0]["name"], "since")
        self.assertEqual(serialized_relationship["properties"][0]["type"], "INTEGER")

        # Assert the graph's description
        self.assertEqual(serialized_graph["description"], "A sample graph schema")
