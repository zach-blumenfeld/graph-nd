import io
import os
import unittest
import warnings
from contextlib import redirect_stdout

from dotenv import load_dotenv
from neo4j import GraphDatabase
from graph_nd import GraphRAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class TestGraphSchemaFromCompaniesGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress warnings - there are many produced with this schema :(
        warnings.filterwarnings("ignore")

        # Connect to the companies database
        uri = "neo4j+s://demo.neo4jlabs.com"
        username = "companies"
        password = "companies"

        cls.db_client = GraphDatabase.driver(uri, auth=(username, password))
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")
        cls.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        cls.embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

        # Create GraphRAG instance
        cls.graphrag = GraphRAG(cls.db_client, cls.llm, cls.embedding_model)

        # Create schema with specific text embedding index map
        cls.graphrag.schema.from_existing_graph(text_embed_index_map={'news': 'text'})

    @classmethod
    def tearDownClass(cls):
        cls.db_client.close()

    def test_schema_with_text_embedding_and_fulltext(self):
        """Test that the schema created with text_embed_index_map={'news': 'text'} is valid"""
        schema = self.graphrag.schema.schema

        # Verify basic schema structure
        self.assertIsNotNone(schema, "Schema should be created")
        self.assertEqual(len(schema.nodes), 8, "Schema should have 8 node definitions")
        self.assertEqual(len(schema.relationships), 13, "Schema should have 13 relationship definitions")

        # Verify Chunk node
        chunk_node = next((node for node in schema.nodes if node.label == "Chunk"), None)
        self.assertIsNotNone(chunk_node, "Schema should include Chunk nodes")

        # Find vector search fields
        vector_search_fields = [field for field in chunk_node.searchFields
                                if field.type == "TEXT_EMBEDDING"]

        # Verify there is one vector search field
        self.assertEqual(len(vector_search_fields), 1,
                           "Chunk node should have one vector search field")

        # 3. Verify the vector search field is correctly configured
        vector_field = vector_search_fields[0]
        self.assertEqual(vector_field.calculatedFrom, "text",
                         "Vector search field should be calculated from 'text' property")

        # Find fulltext search fields
        fulltext_search_fields = [field for field in chunk_node.searchFields
                                if field.type == "FULLTEXT"]

        # Verify there is one fulltext search field
        self.assertEqual(len(fulltext_search_fields), 1,
                           "Chunk node should have one fulltext search field")

        # 3. Verify the vector search field is correctly configured
        fulltext_field = fulltext_search_fields[0]
        self.assertEqual(fulltext_field.calculatedFrom, "text",
                         "fulltext search field should be calculated from 'text' property")


    def test_agent_with_vector_search(self):
        """Test that the agent can use the vector search capability without errors"""
        # Capture stdout to check if anything was printed
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # Just verify it runs without error
                self.graphrag.agent("What articles discuss technology innovations? Use semantic search.")
                # If we get here without exception, the test passes
            except Exception as e:
                self.fail(f"Vector search query raised an exception: {str(e)}")

        # Verify that something was printed (not empty response)
        output = f.getvalue()
        self.assertTrue(len(output) > 0, "Agent should print some output")
        print("Agent executed query with vector search")

    def test_agent_with_fulltext_search(self):
        """Test that the agent can use the fulltext search capability without errors"""
        # Capture stdout to check if anything was printed
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # Just verify it runs without error
                self.graphrag.agent("What chunks discuss GPUs?. use fulltext search")
                # If we get here without exception, the test passes
            except Exception as e:
                self.fail(f"fulltext search query raised an exception: {str(e)}")

        # Verify that something was printed (not empty response)
        output = f.getvalue()
        self.assertTrue(len(output) > 0, "Agent should print some output")
        print("Agent executed query with fulltext search")

    def test_agent_with_aggregation(self):
        """Test that the agent can use aggregation without errors"""
        # Capture stdout to check if anything was printed
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # Just verify it runs without error
                self.graphrag.agent("How many people exist who have been both a board member and CEO at some point? even of separate orgs?")
                # If we get here without exception, the test passes
            except Exception as e:
                self.fail(f"aggregation query raised an exception: {str(e)}")

        # Verify that something was printed (not empty response)
        output = f.getvalue()
        self.assertTrue(len(output) > 0, "Agent should print some output")
        print("Agent executed aggregation query")



if __name__ == "__main__":
    unittest.main()
