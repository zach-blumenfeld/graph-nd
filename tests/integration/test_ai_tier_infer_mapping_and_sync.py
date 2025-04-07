import unittest
import os
from pprint import pprint
from unittest.mock import Mock

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase

from graph_nd import GraphRAG, AITier
from snowflake.snowpark import Session

from graph_nd.ai_tier.ai_tier import Models
from graph_nd.ai_tier.data_source import SnowflakeDB, SourceMappingDirectives, SourceMappings
from graph_nd.graphrag.graph_schema import GraphSchema
from graph_nd.graphrag.utils import run_async_function


# noinspection SqlNoDataSourceInspection
class TestAITierInferMappingAndSync(unittest.TestCase):
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
        # Set up embedding model (e.g., OpenAI)
        embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

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


        # Initialize AITier
        cls.ai_tier = AITier(
            models = Models(llm=llm, embedder=embedding_model),
            knowledge_graph=cls.db_client,
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
        session.sql("CREATE TABLE IF NOT EXISTS Orders (OrderID INT, ProductID INT, CustomerID INT, Quantity INT)").collect()
        session.sql("""
            INSERT INTO Orders (OrderID, ProductID, CustomerID, Quantity) VALUES
            (101, 1, 1, 10),
            (102, 2, 1, 20),
            (103, 3, 2, 15),
            (104, 3, 3, 10),
            (105, 1, 3, 5)
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
                ) COMMENT='This table shows what products (ParentID) are dependent on which components (ComponentID)'
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
        session.sql("CREATE TABLE IF NOT EXISTS Components (ComponentID INT, Name STRING) COMMENT='This table shows what Components (ComponentID) are dependent on which Suppliers (SupplierID)'").collect()
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

    def test_sync(self):

        # infer mapping
        use_case = "which customers depend on which suppliers?"
        self.ai_tier.knowledge.infer_mapping(use_case)
        print(self.ai_tier.knowledge.mappings.model_dump_json(indent=4))

        #sync
        self.ai_tier.knowledge.sync()

        graph_schema = self.ai_tier.knowledge.graphrag.schema.schema

        # Validate the inferred schema
        self.assertIsInstance(graph_schema, GraphSchema)


    @classmethod
    def tearDownClass(cls):
        """
        Clean up the Snowflake data sources and neo4j after testing.
        """
        # Drop databases
        cls.session_product.sql("DROP DATABASE IF EXISTS PRODUCTS_DB").collect()
        cls.session_bom.sql("DROP DATABASE IF EXISTS BOM_DB").collect()

        # Close Snowflake sessions
        cls.session_product.close()
        cls.session_bom.close()
        #
        # #drop Neo4j data
        # with cls.db_client.session() as session:
        #     # Delete all nodes and relationships
        #     session.run("MATCH (n) DETACH DELETE n")
        #
        #     # Retrieve and drop specific indexes
        #     result = session.run("""
        #         SHOW INDEXES YIELD name, type
        #         WHERE type IN ["FULLTEXT", "VECTOR"]
        #         RETURN name
        #     """)
        #
        #     for record in result:
        #         index_name = record["name"]
        #         session.run(f"DROP INDEX {index_name} IF EXISTS")
        #
        #     #drop constraints
        #     session.run("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")
        #
        # cls.db_client.close()


if __name__ == "__main__":
    unittest.main()
