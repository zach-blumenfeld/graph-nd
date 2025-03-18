import os
import unittest

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from Graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, GraphSchema
from graphrag import GraphRAG
from table_mapping import NodeTableMapping
from utils import format_table_as_markdown_preview


class TestGraphRAGNodeRelTableMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup a GraphRAG instance and define a base schema.
        """
        # Initialize LLM
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        # Initialize GraphRAG
        cls.graphrag = GraphRAG(db_client=None, llm=llm)
        #define schema
        person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="person id"),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="person name"),
                PropertySchema(name="age", type="INTEGER", description="person age in years"),
                PropertySchema(name="bio", type="STRING", description="person's biography")
            ],
            description="A Person with text embedding on bio"
        )
        movie_schema = NodeSchema(
            id=PropertySchema(name="movie_id", type="STRING", description="movie id"),
            label="Movie",
            properties=[
                PropertySchema(name="title", type="STRING", description="The movie title"),
                PropertySchema(name="release_year", type="INTEGER", description="The movie release year"),
            ],
            description="A Movie with fulltext index on the title"
        )
        acted_in_schema = RelationshipSchema(
            type="ACTED_IN",
            id=None,  # No unique identifier for relationships in this test
            properties=[
                PropertySchema(name="role", type="STRING", description="The role played by the person in the movie.")
            ],
            queryPatterns=[QueryPattern(startNode="Person", endNode="Movie", description="person acted in movie")],
            description="a person acting in a movie"
        )
        # Define the schema (using structure from test_graph_data_merge_search_fields)
        cls.graphrag.schema.define(GraphSchema(nodes=[person_schema, movie_schema],
                                               relationships=[acted_in_schema]))

    def test_node_mapping(self):
        """
        Test node table mapping.
        """
        # Table name and input data
        table_name = "actors"
        table_header = ["id", "names", "age (in years)"]
        table_rows = [
            [1, "Alice",30],
            [2, "Bob", 25],
            [3, "Charlie", 35]
        ]
        table_preview = format_table_as_markdown_preview(table_header, table_rows)

        # Call the method
        node_mapping = self.graphrag.data.get_table_node_mapping(table_name, table_preview)
        # Assert individual fields of the node mapping
        self.assertEqual(node_mapping.tableName, 'actors', "The table name does not match.")
        self.assertEqual(node_mapping.nodeLabel, 'Person', "The node label does not match.")
        self.assertEqual(node_mapping.nodeId.propertyName, 'person_id', "The node id does not match.")
        self.assertEqual(node_mapping.nodeId.columnName, 'id', "The node id column does not match.")
        # Expected mappings as a set of tuples
        expected_properties = {("names", "name"), ("age (in years)", "age")}
        # Convert the list of PropertyMapping objects to a set of tuples
        actual_properties = {
            (prop.columnName, prop.propertyName) for prop in node_mapping.properties
        }
        # Assert equality of the sets
        self.assertEqual(
            actual_properties,
            expected_properties,
            "The node properties mapping does not match."
        )

    def test_relationships_mapping(self):
        """
        Test relationship table mapping.
        """
        # Table name and input data
        table_name = "roles"
        table_header = ["actorIds", "movieIds", "actorRoles"]
        table_rows = [
            [1, "M101", "Protagonist"],
            [2, "M101", "Villain"],
            [2, "M102", "Hacker"],
        ]
        table_preview = format_table_as_markdown_preview(table_header, table_rows)
        # Call the method
        rels_mapping = self.graphrag.data.get_table_relationships_mapping(table_name, table_preview)
        # Assert fields of the relationship mapping
        self.assertEqual(rels_mapping.tableName, 'roles', "The table name does not match.")

        # Assert first relationship mapping fields
        self.assertEqual(rels_mapping.relationshipMaps[0].relationshipType, 'ACTED_IN',
                         "The relationship type doesn't not match.")
        self.assertIsNone(rels_mapping.relationshipMaps[0].relationshipId,
                          "The relationship id should be None.")
        # Expected mappings as a set of tuples
        expected_rel_properties = {("actorRoles", "role")}
        # Convert the list of PropertyMapping objects to a set of tuples
        actual_rel_properties = {
            (prop.columnName, prop.propertyName) for prop in rels_mapping.relationshipMaps[0].properties
        }
        # Assert equality of the sets
        self.assertEqual(
            actual_rel_properties,
            expected_rel_properties,
            "The node properties mapping does not match."
        )

       # Assert start node fields
        self.assertEqual(rels_mapping.relationshipMaps[0].startNodeMap.nodeId.columnName, 'actorIds',
                         "The start node id column does not match.")
        self.assertEqual(rels_mapping.relationshipMaps[0].startNodeMap.nodeId.propertyName, 'person_id',
                         "The start node id does not match.")
        self.assertEqual(rels_mapping.relationshipMaps[0].startNodeMap.nodeLabel, 'Person',
                         "The start node label does not match.")
        self.assertIsNone(rels_mapping.relationshipMaps[0].startNodeMap.properties,
                          "The start node properties should be None.")

        # Assert end node fields
        self.assertEqual(rels_mapping.relationshipMaps[0].endNodeMap.nodeId.columnName, 'movieIds',
                         "The end node id column does not match.")
        self.assertEqual(rels_mapping.relationshipMaps[0].endNodeMap.nodeId.propertyName, 'movie_id',
                         "The end node id does not match.")
        self.assertEqual(rels_mapping.relationshipMaps[0].endNodeMap.nodeLabel, 'Movie',
                         "The end node label does not match.")
        self.assertIsNone(rels_mapping.relationshipMaps[0].endNodeMap.properties,
                          "The end node properties should be None.")


    @classmethod
    def tearDownClass(cls):
        """
        Cleanup resources.
        """
        del cls.graphrag
