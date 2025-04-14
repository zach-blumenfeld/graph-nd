import unittest
import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from graph_nd import GraphRAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class TestGraphRAGRelWithKeyMerge(unittest.TestCase):
    PARENT_DIR = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls):
        """Set up test files and initialize objects."""
        schema_file = os.path.join(cls.PARENT_DIR, 'data-models', 'customer-graph.json')

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


        # Initialize GraphRAG
        cls.graphrag = GraphRAG(db_client=cls.db_client)

        # Load the schema
        cls.graphrag.schema.load(schema_file)

        # load data
        order_df = pd.read_csv(os.path.join(cls.PARENT_DIR,"data", "order-details.csv"))
        contains_records = (order_df[['orderId', 'articleId', 'txId', 'price']]).rename(
            columns={'orderId': 'start_node_id', 'articleId': 'end_node_id'}).to_dict(orient="records")
        cls.graphrag.data.merge_relationships(rel_type='CONTAINS',
                                          start_node_label='Order',
                                          end_node_label='Article',
                                          records=contains_records)


    @classmethod
    def tearDownClass(cls):
        # Clean database
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

    def test_relationship_count(self):
        rel_pattern_counts = [{"cnt": 23199,"startNode": ["Order"],"type": "CONTAINS","endNode": ["Article"]}]
        rel_res = self.db_client.execute_query(
            query_="""
            MATCH (n)-[r]->(m)
            RETURN count(*) AS cnt, labels(n) AS startNode, type(r) AS type, labels(m) AS endNode
            ORDER BY type
            """,
            result_transformer_ = lambda r: r.data()
        )
        self.assertEqual(rel_res, rel_pattern_counts)

    def test_validate_constraints(self):
        expected_constraint_results = [
              {
                "labelsOrTypes": ["Article"],
                "properties": ["articleId"]
              },
              {
                "labelsOrTypes": ["Order"],
                "properties": ["orderId"]
              }
        ]
        constraint_res = self.db_client.execute_query(
            query_='SHOW CONSTRAINTS YIELD labelsOrTypes, properties ORDER BY labelsOrTypes',
            result_transformer_ = lambda r: r.data()
        )
        self.assertEqual(constraint_res, expected_constraint_results)
