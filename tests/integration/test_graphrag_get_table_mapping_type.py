import os
import unittest

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, GraphSchema
from graphrag import GraphRAG
from utils import format_table_as_markdown_preview


class TestGraphRAGDataGetTableMappingType(unittest.TestCase):
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

    def test_get_table_mapping_type_single_node(self):
        """
        Test SINGLE_NODE mapping type using NodeTable schema.
        """
        # Table name and input data
        table_name = "actors"
        table_header = ["id", "name", "age"]
        table_rows = [
            [1, "Alice",30],
            [2, "Bob", 25],
            [3, "Charlie", 35]
        ]
        table_preview = format_table_as_markdown_preview(table_header, table_rows)

        # Call the method
        table_type = self.graphrag.data.get_table_mapping_type(table_name, table_preview)

        # Assert the mapping type is SINGLE_NODE
        self.assertEqual(table_type, "SINGLE_NODE", "The table mapping type should be SINGLE_NODE.")

    def test_get_table_mapping_type_relationships(self):
        """
        Test RELATIONSHIPS mapping type using RelationshipsTable schema.
        """
        # Table name and input data
        table_name = "roles"
        table_header = ["person_id", "movie_id", "role"]
        table_rows = [
            [1, "M101", "Protagonist"],
            [2, "M101", "Villain"],
            [2, "M102", "Hacker"],
        ]
        table_preview = format_table_as_markdown_preview(table_header, table_rows)
        # Call the method
        table_type = self.graphrag.data.get_table_mapping_type(table_name, table_preview)

        # Assert the mapping type is RELATIONSHIPS
        self.assertEqual(table_type, "RELATIONSHIPS", "The table mapping type should be RELATIONSHIPS.")

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup resources.
        """
        del cls.graphrag
