import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from snowflake.snowpark import Session
from typing import List, Dict

from data_source import SnowflakeDB

# Load environment variables from a .env file
load_dotenv()


class TestSnowflakeDB(unittest.TestCase):
    """
    Integration tests for the SnowflakeDB class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a Snowflake session, create test database, and seed data.
        """
        # Snowflake connection parameters
        cls.params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "database": "TEST_DB",
            "schema": "PUBLIC",
        }

        # Create session
        cls.session = Session.builder.configs(cls.params).create()

        # Create a test database
        cls.session.sql("CREATE OR REPLACE DATABASE TEST_DB").collect()

        # Set a comment on the PUBLIC schema
        cls.schema_comment = "This is the primary testing database."
        cls.session.sql(f"COMMENT ON SCHEMA PUBLIC IS '{cls.schema_comment}'").collect()

        # Initialize SnowflakeDB instance for testing
        cls.snowflake_db = SnowflakeDB(**cls.params)

        # Seed test data
        cls.seed_test_data()

    @classmethod
    def seed_test_data(cls):
        """
        Create tables and seed sample data for a retail domain.
        """
        # Write Customers table data and automatically create the table
        cls.session.write_pandas(
            pd.DataFrame({
                "CUSTOMER_ID": [1, 2, 3],
                "CUSTOMER_NAME": ["Alice", "Bob", "Charlie"],
                "EMAIL": ["alice@example.com", "bob@example.com", "charlie@example.com"]
            }),
            table_name="CUSTOMERS",
            auto_create_table=True,
            overwrite=True
        )

        # Add a primary key to the CUSTOMERS table
        cls.session.sql("ALTER TABLE CUSTOMERS ADD CONSTRAINT pk_customers PRIMARY KEY (customer_id)").collect()
        # Add a comment to the CUSTOMERS table
        cls.customer_table_comment = "Customer Information"
        cls.session.sql(f"COMMENT ON TABLE CUSTOMERS IS '{cls.customer_table_comment}'").collect()

        # Write Products table data and automatically create the table
        cls.session.write_pandas(
            pd.DataFrame({
                "PRODUCT_ID": [101, 102, 103],
                "PRODUCT_NAME": ["Laptop", "Phone", "Tablet"],
                "PRICE": [1000.0, 500.0, 300.0]
            }),
            table_name="PRODUCTS",
            auto_create_table=True,
            overwrite=True
        )

        # Add a primary key to the PRODUCTS table
        cls.session.sql("ALTER TABLE PRODUCTS ADD CONSTRAINT pk_products PRIMARY KEY (product_id)").collect()

        # Write Purchases table data and automatically create the table
        cls.session.write_pandas(
            pd.DataFrame({
                "PURCHASE_ID": [1, 2, 3],
                "CUSTOMER_ID": [1, 2, 3],
                "PRODUCT_ID": [101, 102, 103],
                "PURCHASE_DATE": ["2023-10-01", "2023-10-02", "2023-10-03"]
            }),
            table_name="PURCHASES",
            auto_create_table=True,
            overwrite=True
        )

        # Add a primary key to the PURCHASES table
        cls.session.sql("ALTER TABLE PURCHASES ADD CONSTRAINT pk_purchases PRIMARY KEY (purchase_id)").collect()

        # Add foreign key constraints to the PURCHASES table
        cls.session.sql("""
            ALTER TABLE PURCHASES ADD CONSTRAINT fk_customer_id
            FOREIGN KEY (customer_id) REFERENCES CUSTOMERS (customer_id)
        """).collect()

        cls.session.sql("""
            ALTER TABLE PURCHASES ADD CONSTRAINT fk_product_id
            FOREIGN KEY (product_id) REFERENCES PRODUCTS (product_id)
        """).collect()

        # Add comments to the foreign key columns in the PURCHASES table
        cls.customer_id_fk_comment = "Foreign key: References CUSTOMERS.customer_id"
        cls.session.sql(f"COMMENT ON COLUMN PURCHASES.customer_id IS '{cls.customer_id_fk_comment}'").collect()
        cls.product_id_fk_comment = "Foreign key: References PRODUCTS.product_id"
        cls.session.sql(f"COMMENT ON COLUMN PURCHASES.product_id IS '{cls.product_id_fk_comment}'").collect()

    def test_schema(self):
        """
        Test that the schema() method correctly retrieves database and table metadata.
        """
        # Step 1: Call the schema() method and retrieve database metadata
        schema_info = self.snowflake_db.schema()

        # Step 2: Validate top-level Database Metadata
        self.assertEqual(schema_info.name, "TEST_DB.PUBLIC")  # Replace with expected database name
        self.assertEqual(
            schema_info.description,
            "This is the primary testing database."  # Replace with expected description
        )

        # Step 3: Validate Table Entities Metadata
        self.assertEqual(len(schema_info.entities), 3)  # Should find 3 tables
        table_names = [entity.name for entity in schema_info.entities]
        self.assertIn("CUSTOMERS", table_names)
        self.assertIn("PRODUCTS", table_names)
        self.assertIn("PURCHASES", table_names)

        # Step 4: Validate Specific Table Metadata
        for table in schema_info.entities:
            if table.name == "CUSTOMERS":
                self.assertEqual(table.description, self.customer_table_comment)
                self.assertEqual(len(table.entity_schema), 3)  # Expected number of fields

                # Step 5: Validate Field Metadata
                field_map = {field["name"]: field for field in table.entity_schema}
                self.assertIn("CUSTOMER_ID", field_map)
                self.assertIn("CUSTOMER_NAME", field_map)
                self.assertIn("EMAIL", field_map)

                # Validate individual field properties
                self.assertEqual(field_map["CUSTOMER_ID"]["primary key"], 'Y')
                self.assertEqual(field_map["CUSTOMER_ID"]["type"], 'NUMBER(38,0)')
                self.assertEqual(field_map["CUSTOMER_NAME"]["primary key"], 'N')
                self.assertEqual(field_map["CUSTOMER_NAME"]["type"], "VARCHAR(16777216)")
                self.assertEqual(field_map["EMAIL"]["primary key"], 'N')
                self.assertEqual(field_map["EMAIL"]["type"], "VARCHAR(16777216)")

            if table.name == "PRODUCTS":
                self.assertEqual(table.description, '')
                self.assertEqual(len(table.entity_schema), 3)  # Expected number of fields

                field_map = {field["name"]: field for field in table.entity_schema}
                self.assertIn("PRODUCT_ID", field_map)
                self.assertIn("PRODUCT_NAME", field_map)
                self.assertIn("PRICE", field_map)

            if table.name == "PURCHASES":
                self.assertEqual(len(table.entity_schema), 4)  # Expected number of fields

                field_map = {field["name"]: field for field in table.entity_schema}
                self.assertIn("PURCHASE_ID", field_map)
                self.assertIn("CUSTOMER_ID", field_map)
                self.assertIn("PRODUCT_ID", field_map)
                self.assertIn("PURCHASE_DATE", field_map)

                self.assertEqual(field_map["PRODUCT_ID"]["comment"], self.product_id_fk_comment)
                self.assertEqual(field_map["CUSTOMER_ID"]["comment"], self.customer_id_fk_comment)

    def test_get_table(self):
        """
        Test the `get_table` method to fetch table data.
        """
        # Test Customers table
        customers = self.snowflake_db.get_table("customers")
        self.assertEqual(len(customers), 3)
        self.assertEqual(customers[0]["CUSTOMER_NAME"], "Alice")

        # Test Products table
        products = self.snowflake_db.get_table("products")
        self.assertEqual(len(products), 3)
        self.assertEqual(products[1]["PRODUCT_NAME"], "Phone")
        self.assertEqual(products[1]["PRICE"], 500.0)

        # Test Purchases table
        purchases = self.snowflake_db.get_table("purchases")
        self.assertEqual(len(purchases), 3)
        self.assertEqual(purchases[2]["PURCHASE_DATE"], "2023-10-03")

    @classmethod
    def tearDownClass(cls):
        """
        Tear down the test database and close the session.
        """
        cls.session.sql("DROP DATABASE IF EXISTS TEST_DB").collect()
        cls.snowflake_db.close()


if __name__ == "__main__":
    unittest.main()
