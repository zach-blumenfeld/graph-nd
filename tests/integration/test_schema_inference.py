import unittest
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph_nd.graphrag.graph_schema import GraphSchema
from graph_nd import GraphRAG  # Replace with actual import path


class TestGraphRAGSchemaInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the real LLM and database client before running tests.
        """
        # Set up LLM API (e.g., OpenAI)
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")

        cls.real_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        cls.mock_db_client = None  # Replace with a real or mock database client
        cls.graph_rag = GraphRAG(cls.mock_db_client, cls.real_llm)


    def test_infer_with_real_llm(self):
        """
        Test the infer method using a real LLM with a description.
        """
        description = "A graph schema representing people and movies, where people act in movies."
        inferred_schema = self.graph_rag.schema.infer(description)

        print("Result from infer:")
        print(inferred_schema)

        # Assertions to validate schema integrity
        self.assertIsInstance(inferred_schema, GraphSchema)
        self.assertGreater(len(inferred_schema.nodes), 1)  # At least 2 nodes inferred
        self.assertGreater(len(inferred_schema.relationships), 0)  # At least one relationship inferred

    def test_infer_from_sample_with_real_llm(self):
        """
        Test the infer_from_sample method using a real LLM with sample data.
        """
        sample_data = """
        Leonardo DiCaprio (id: 'imdb-a123') played the lead role in Inception (id: 'imdb-m321') which was released in 2010.
        """
        inferred_schema = self.graph_rag.schema.infer_from_sample(sample_data)

        print("Result from infer_from_sample:")
        print(inferred_schema)

        # Assertions
        self.assertIsInstance(inferred_schema, GraphSchema)
        self.assertGreater(len(inferred_schema.nodes), 1)
        self.assertGreater(len(inferred_schema.relationships), 0)

    def test_craft_from_dict_with_real_llm(self):
        """
        Test the craft_from_dict method using a real LLM with a JSON-like definition.
        """
        schema_dict = '{"nodes": [{"label": "Person"}, {"label": "Movie"}], "relationships": [{"type": "ACTS_IN"}]}'
        crafted_schema = self.graph_rag.schema.craft_from_dict(schema_dict)

        print("Result from craft_from_dict:")
        print(crafted_schema)

        # Assertions
        self.assertIsInstance(crafted_schema, GraphSchema)
        self.assertGreater(len(crafted_schema.nodes), 1)
        self.assertGreater(len(crafted_schema.relationships), 0)


if __name__ == "__main__":
    unittest.main()
