import unittest

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph_nd import GraphRAG, SubSchema
import os


class TestGraphRAGMergePDF(unittest.TestCase):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def setUpClass(cls):
        """Set up test files and initialize objects."""
        cls.pdf_file = os.path.join(cls.DATA_DIR, "credit-notes.pdf")

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

        # Load the schema
        cls.graphrag.schema.load(os.path.join(cls.DATA_DIR, "customer-graph.json"))

    def test_merge_pdf(self):
        # Verify merge_pdf integration
        for i in range(2): #less than 1% of the time the llm will miss nodes/rels. This is random so repeating twice ensure we get everything
            self.graphrag.data.merge_pdf(self.pdf_file, nodes_only=False,
                                    sub_schema=SubSchema(patterns=[
                                        ('CreditNote', 'REFUND_FOR_ORDER', 'Order'),
                                        ('CreditNote', "REFUND_OF_ARTICLE", 'Article')]
                                        )
                                    )

        #ensure node label counts
        node_label_counts = [
            {
                "cnt": 118,
                "nodeLabels": ["Article"]
            },
            {
                "cnt": 307,
                "nodeLabels": ["CreditNote"]
            },
            {
                "cnt": 242,
                "nodeLabels": [ "Order"]
            },
            {
                "cnt": 2,
                "nodeLabels": ["__Source__"]
            }
        ]
        node_res = self.db_client.execute_query(
            query_="""
            MATCH (n)
            RETURN count(*) AS cnt, labels(n) AS nodeLabels
            ORDER BY nodeLabels
            """,
            result_transformer_ = lambda r: r.data()
        )
        self.assertEqual(node_res, node_label_counts)

        #ensure relationship pattern count
        rel_pattern_counts = [
            {
                "cnt": 307,
                "startNode": ["CreditNote"],
                "type": "REFUND_FOR_ORDER",
                "endNode": ["Order"],
            },
            {
                "cnt": 307,
                "startNode": ["CreditNote"],
                "type": "REFUND_OF_ARTICLE",
                "endNode": ["Article"],
            },
        ]
        rel_res = self.db_client.execute_query(
            query_="""
            MATCH (n)-[r]->(m)
            RETURN count(*) AS cnt, labels(n) AS startNode, type(r) AS type, labels(m) AS endNode
            ORDER BY type
            """,
            result_transformer_ = lambda r: r.data()
        )
        self.assertEqual(rel_res, rel_pattern_counts)


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