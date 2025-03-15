import unittest
import json
from tempfile import NamedTemporaryFile
from GraphSchema import GraphSchema, NodeSchema, RelationshipSchema, PropertySchema, QueryPattern
from graphrag import GraphRAG  # Replace with actual import for GraphRAG


class TestGraphSchemaExportAndLoad(unittest.TestCase):
    def setUp(self):
        """
        Set up a test instance of GraphRAG.Schema and a sample graph schema.
        """
        # Mock db_client and llm
        db_client = None  # Replace with actual or mock db_client if needed
        llm = None  # Replace with actual or mock llm if needed

        # Instantiate the Schema manager
        self.schema_manager = GraphRAG.Schema(db_client, llm)

        # Create a sample graph schema (using the example above)
        self.graph_schema = GraphSchema(
            description="A graph schema representing people, movies, and awards",
            nodes=[
                NodeSchema(
                    id=PropertySchema(name="id", type="STRING", description="Person unique identifier"),
                    label="Person",
                    properties=[
                        PropertySchema(name="name", type="STRING", description="Full name of the person"),
                        PropertySchema(name="age", type="INTEGER", description="Age of the person")
                    ],
                    description="Person entity representing individuals in the graph"
                ),
                NodeSchema(
                    id=PropertySchema(name="id", type="STRING", description="Movie unique identifier"),
                    label="Movie",
                    properties=[
                        PropertySchema(name="title", type="STRING", description="Title of the movie"),
                        PropertySchema(name="release_year", type="INTEGER", description="Year the movie was released")
                    ],
                    description="Movie entity representing films in the graph"
                ),
                NodeSchema(
                    id=PropertySchema(name="id", type="STRING", description="Award unique identifier"),
                    label="Award",
                    properties=[
                        PropertySchema(name="name", type="STRING", description="Name of the award"),
                    ],
                    description="Award entity representing awards given to movies or people"
                )
            ],
            relationships=[
                RelationshipSchema(
                    id=None,
                    type="ACTED_IN",
                    queryPatterns=[QueryPattern(startNode="Person", endNode="Movie",
                                                description="Represent Person acted in a Movie")],
                    properties=[PropertySchema(name="role", type="STRING", description="Role played by the person")],
                    description="Indicates that a Person acted in a Movie"
                ),
                RelationshipSchema(
                    id=None,
                    type="WON",
                    queryPatterns=[QueryPattern(startNode="Person", endNode="Award",
                                                description="Represent a Person won an Award")],
                    properties=[PropertySchema(name="year", type="INTEGER", description="Year the award was won")],
                    description="Indicates that a Person won an Award"
                ),
                RelationshipSchema(
                    id=None,
                    type="DIRECTED",
                    queryPatterns=[QueryPattern(startNode="Person", endNode="Movie",
                                                description="Represent a Person directed a Movie")],
                    properties=[],
                    description="Indicates that a Person directed a Movie"
                )
            ]
        )

        # Assign to self.schema_manager.graph_schema
        self.schema_manager.schema = self.graph_schema

    def test_export_schema(self):
        """
        Test exporting the GraphSchema to a temporary file.
        """
        with NamedTemporaryFile(delete=True) as tmp_file:
            # Export schema
            self.schema_manager.export(tmp_file.name)

            # Read content from file
            with open(tmp_file.name, 'r') as file:
                data = json.load(file)

                # Validate exported content (Non-exhaustive, add more checks as needed)
                self.assertEqual(data["description"], "A graph schema representing people, movies, and awards")
                self.assertEqual(len(data["nodes"]), 3)  # 3 Nodes: Person, Movie, Award
                self.assertEqual(len(data["relationships"]), 3)  # 3 Relationships
                self.assertIn("ACTED_IN", [rel["type"] for rel in data["relationships"]])

    def test_load_schema(self):
        """
        Test loading the GraphSchema from a temporary file.
        """
        with NamedTemporaryFile(delete=True, mode='w+') as tmp_file:
            # Dump the graph schema to the file
            json.dump(self.graph_schema.model_dump(), tmp_file, indent=4)
            tmp_file.seek(0)  # Reset file pointer to the beginning

            # Load schema
            self.schema_manager.load(tmp_file.name)

            # Verify that the loaded schema matches the original
            loaded_schema = self.schema_manager.schema
            self.assertEqual(loaded_schema.description, self.graph_schema.description)
            self.assertEqual(len(loaded_schema.nodes), len(self.graph_schema.nodes))
            self.assertEqual(len(loaded_schema.relationships), len(self.graph_schema.relationships))

    def test_export_without_schema(self):
        """
        Test exporting when no schema is defined.
        """
        self.schema_manager.schema = None

        with self.assertRaises(ValueError) as context:
            self.schema_manager.export("dummy_path.json")

        self.assertEqual(str(context.exception), "[Schema] No schema defined to export.")


if __name__ == "__main__":
    unittest.main()
