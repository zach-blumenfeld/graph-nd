import os
import unittest
from collections import Counter
from pprint import pprint

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from neo4j import GraphDatabase

from Graph_schema import GraphSchema
from graphrag import GraphRAG


class TestMergeCSVsFunctional(unittest.TestCase):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def setUpClass(cls):
        """Set up test files and initialize objects."""
        cls.csv_files = [
            os.path.join(cls.DATA_DIR, "components.csv"),
            os.path.join(cls.DATA_DIR, "component-types.csv"),
            os.path.join(cls.DATA_DIR, "component-input-output.csv"),
        ]
        # Load environment variables from a .env file
        load_dotenv()

        # Retrieve and check each required variable individually
        NEO4J_URI = os.getenv("NEO4J_URI")
        if not NEO4J_URI:
            raise EnvironmentError(
                "Environment variable 'NEO4J_URI' is not set. Please configure it for integration tests.")

        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        if not NEO4J_USERNAME:
            raise EnvironmentError(
                "Environment variable 'NEO4J_USERNAME' is not set. Please configure it for integration tests.")

        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        if not NEO4J_PASSWORD:
            raise EnvironmentError(
                "Environment variable 'NEO4J_PASSWORD' is not set. Please configure it for integration tests.")

        # Create the database client if all environment variables are present
        cls.db_client = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # Set up embedding model (e.g., OpenAI)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")
        embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        # Initialize GraphRAG
        cls.graphrag = GraphRAG(db_client=cls.db_client, llm=llm, embedding_model=embedding_model)

    def test_merge_csvs(self):
        """Test merging of CSV files and count unique nodes + relationships."""


        # Infer schema
        self.graphrag.schema.infer(''''
        a simple graph of hardware components where components can be types of or inputs to other components
        ''')
        pprint(f'Inferred schema:\n {self.graphrag.schema.schema.prompt_str()}')

        # Count node labels
        self.assertEqual(len(self.graphrag.schema.schema.nodes), 1, "Expected one node label in the schema")

        # Count relationship types
        self.assertEqual(len(self.graphrag.schema.schema.relationships), 2, "Expected two relationship types in the schema")

        # Call the merge_csvs method
        self.graphrag.data.merge_csvs(file_paths=self.csv_files)

        # count unique nodes should equal number of rows in components.csv
        with self.db_client.session() as session:
            # Query the number of unique nodes
            unique_node_count = session.run("MATCH (n) RETURN count(n) AS node_count").single()["node_count"]

        # Expected node count is the number of rows in components.csv
        expected_node_count = sum(1 for _ in open(self.csv_files[0])) - 1  # Subtract 1 for the header row
        self.assertEqual(unique_node_count, expected_node_count,
                         "Number of unique nodes does not match rows in components.csv")

        # count unique relationships should equal number of rows in component-types.csv + components-input-outpus.csv
        with self.db_client.session() as session:
            # Query the number of unique relationships
            unique_relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS relationship_count").single()[
                "relationship_count"]

        # Expected relationship count is the sum of rows in component-types.csv and component-input-output.csv
        expected_relationship_count = (
                sum(1 for _ in open(self.csv_files[1])) - 1  # Rows in component-types.csv (subtract 1 for header)
                + sum(1 for _ in open(self.csv_files[2])) - 1
        # Rows in component-input-output.csv (subtract 1 for header)
        )
        self.assertEqual(unique_relationship_count, expected_relationship_count,
                         "Number of relationships does not match expected relationships")

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup resources.
        """

        with cls.db_client.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

            # Retrieve and drop specific indexes
            result = session.run("""
                SHOW INDEXES YIELD name, type
                WHERE type IN ["FULLTEXT", "VECTOR"]
                RETURN name
            """)

            for record in result:
                index_name = record["name"]
                session.run(f"DROP INDEX {index_name} IF EXISTS")

            #drop constraints
            session.run("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")

        cls.db_client.close()
        del cls.graphrag



