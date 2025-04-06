import unittest
import os
from unittest.mock import Mock

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from graph_nd import GraphRAG, AITier
from snowflake.snowpark import Session  # Replace with actual Snowflake connection class

from graph_nd.ai_tier.ai_tier import Models
from graph_nd.ai_tier.data_source import SnowflakeDB
from graph_nd.graphrag.graph_schema import GraphSchema  # Replace with actual import


# noinspection SqlNoDataSourceInspection
class TestAITierInferGraphSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up LLM and Snowflake data sources for integration testing.
        """
        # Load API keys from environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")

        # Initialize OpenAI-compatible LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        # Base connection parameters from environment
        base_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }

        # Connection parameters for Products database
        product_db_params = {
            **base_params,
            "database": "PRODUCTS_DB",
            "schema": "PUBLIC",
        }

        # Connection parameters for BoM database
        bom_db_params = {
            **base_params,
            "database": "BOM_DB",
            "schema": "PUBLIC",
        }

        # Initialize sessions for Products and Bill of Materials databases
        cls.session_product = Session.builder.configs(product_db_params).create()
        cls.session_bom = Session.builder.configs(bom_db_params).create()

        # Seed both databases with multiple tables
        cls.seed_product_data(cls.session_product)
        cls.seed_bom_data(cls.session_bom)

        # Create Data Sources
        snow_product_ds = SnowflakeDB(**product_db_params)
        snow_bom_ds = SnowflakeDB(**bom_db_params)


        # Set up the GraphRAG integration with LLM
        mock_kg_client = Mock()

        # Initialize AITier
        cls.ai_tier = AITier(
            models = Models(llm=llm, embedder=None),
            knowledge_graph=mock_kg_client,
            data_sources=[snow_product_ds, snow_bom_ds],
        )


    @staticmethod
    def seed_product_data(session):
        """
        Seed data for the Products database with multiple related tables.
        """
        # Create Database
        session.sql("CREATE OR REPLACE DATABASE PRODUCTS_DB").collect()

        # Create Products table
        session.sql("CREATE TABLE IF NOT EXISTS Products (ProductID INT, Name STRING, Category STRING)").collect()
        session.sql("""
            INSERT INTO Products (ProductID, Name, Category) VALUES
            (1, 'Widget A', 'Hardware'),
            (2, 'Widget B', 'Hardware'),
            (3, 'Gadget C', 'Electronics')
        """).collect()

        # Create Orders table
        session.sql("CREATE TABLE IF NOT EXISTS Orders (OrderID INT, ProductID INT, Quantity INT)").collect()
        session.sql("""
            INSERT INTO Orders (OrderID, ProductID, Quantity) VALUES
            (101, 1, 10),
            (102, 2, 20),
            (103, 3, 15)
        """).collect()

        # Create Customers table
        session.sql("CREATE TABLE IF NOT EXISTS Customers (CustomerID INT, Name STRING, City STRING)").collect()
        session.sql("""
            INSERT INTO Customers (CustomerID, Name, City) VALUES
            (1, 'Alice', 'New York'),
            (2, 'Bob', 'San Francisco'),
            (3, 'Charlie', 'Los Angeles')
        """).collect()

    @staticmethod
    def seed_bom_data(session):
        """
        Seed data for the Bill of Materials (BoM) database with multiple related tables.
        """
        # Create Database
        session.sql("CREATE OR REPLACE DATABASE BOM_DB").collect()

        # Create BillOfMaterials table
        session.sql("""
                CREATE TABLE IF NOT EXISTS BillOfMaterials (
                    ParentID INT COMMENT 'Same as ProductID in Product domain',
                    ComponentID INT,
                    Quantity FLOAT
                )
            """).collect()

        session.sql("""
            INSERT INTO BillOfMaterials (ParentID, ComponentID, Quantity) VALUES
            (1, 101, 2),  -- Widget A uses these components
            (1, 102, 1),
            (2, 103, 3),  -- Widget B uses these components
            (3, 104, 1),  -- Gadget C uses these components
            (3, 105, 5)
        """).collect()

        # Create Components table
        session.sql("CREATE TABLE IF NOT EXISTS Components (ComponentID INT, Name STRING)").collect()
        session.sql("""
            INSERT INTO Components (ComponentID, Name) VALUES
            (101, 'Screw'),
            (102, 'Bolt'),
            (103, 'Nut'),
            (104, 'Circuit Board'),
            (105, 'Chip')
        """).collect()

        # Create Suppliers table
        session.sql(
            "CREATE TABLE IF NOT EXISTS Suppliers (SupplierID INT, ComponentID INT, SupplierName STRING)").collect()
        session.sql("""
            INSERT INTO Suppliers (SupplierID, ComponentID, SupplierName) VALUES
            (1, 101, 'Supplier A'),
            (2, 102, 'Supplier B'),
            (3, 103, 'Supplier C'),
            (4, 104, 'Supplier D'),
            (5, 105, 'Supplier E')
        """).collect()

    def test_infer_graph_schema(self):
        """
        Test inferring a graph schema combining product and BoM data using the LLM.
        """
        # Description combining the two data sources
        use_case = """
        which customers depend on which suppliers?
        """
        # Infer graph schema using the LLM and GraphRAG
        self.ai_tier.knowledge.infer_graph_schema(use_case)
        graph_schema = self.ai_tier.knowledge.graphrag.schema.schema

        # Validate the inferred schema
        self.assertIsInstance(graph_schema, GraphSchema)
        #TODO: Validating the content of the schema is quite qualitsative so will need to thinkon how best to do that.
        # Also the LLM may provide the same effective schema but choose slightly different rel types and node label names.
        # self.assertGreaterEqual(len(inferred_schema.nodes), 10)  # Ensures multiple nodes exist
        # self.assertGreaterEqual(len(inferred_schema.relationships), 6)  # Nodes must be connected with relationships
        # self.assertIn("Products", [node.label for node in inferred_schema.nodes])
        # self.assertIn("Components", [node.label for node in inferred_schema.nodes])
        # self.assertIn("Suppliers", [node.label for node in inferred_schema.nodes])
        # self.assertIn("CONTAIN", [rel.type for rel in inferred_schema.relationships])
        # self.assertIn("SUPPLY", [rel.type for rel in inferred_schema.relationships])

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the Snowflake data sources after testing.
        """
        # Drop databases
        cls.session_product.sql("DROP DATABASE IF EXISTS PRODUCTS_DB").collect()
        cls.session_bom.sql("DROP DATABASE IF EXISTS BOM_DB").collect()

        # Close Snowflake sessions
        cls.session_product.close()
        cls.session_bom.close()


if __name__ == "__main__":
    unittest.main()
