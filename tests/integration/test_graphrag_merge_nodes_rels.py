import os
import unittest

from dotenv import load_dotenv
from neo4j import GraphDatabase

from graphrag import GraphRAG
from graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, GraphSchema


class TestGraphRAGMergeNodeRels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the database connection for integration testing.
        """
        # Load environment variables from `.env` file
        load_dotenv()

        # Get the Neo4j credentials
        NEO4J_URI = os.getenv("NEO4J_URI")
        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

        # Check that all environment variables are present
        if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
            raise EnvironmentError(
                "Failed to load all required environment variables. Make sure NEO4J_URI, NEO4J_USERNAME, "
                "and NEO4J_PASSWORD are set in a `.env` file."
            )

        # Create the database client
        cls.db_client = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # Initialize the GraphRAG instance we will be using for the tests
        cls.graph_rag = GraphRAG(cls.db_client)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the database and close the connection.
        """

        #delete all nodes and relationships
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
            result = session.run("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")



        cls.db_client.close()

    def test_merge_nodes(self):
        """
        Integration test for the `merge_nodes` method.
        """
        # Define the node schema for "Person"
        person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="Unique identifier for a person."),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="The name of the person."),
                PropertySchema(name="age", type="INTEGER", description="The age of the person.")
            ],
            description="A schema representing a person."
        )

        # Define the schema in the GraphRAG instance
        self.graph_rag.schema.define(GraphSchema(
            nodes=[person_schema],
            relationships=[],
            description="a knowledge graph of people acting in movies"
        ))

        # Use `merge_nodes` to load nodes into the database
        records = [
            {"person_id": 1, "name": "Alice", "age": 30},
            {"person_id": 2, "name": "Bob", "age": 25}
        ]
        self.graph_rag.data.merge_nodes(label="Person", records=records)

        # Verify that nodes were added
        def verify_node_count(tx, label):
            result = tx.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            person_count = session.execute_read(verify_node_count, "Person")

        self.assertEqual(person_count, 2, f"Expected 2 Person nodes, but found {person_count}.")

    def test_merge_relationships(self):
        """
        Integration test for the `merge_relationships` method.
        """
        # Define schema for "Person"
        person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="Unique identifier for a person."),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="The name of the person."),
            ],
            description="A schema representing a person."
        )

        # Define schema for "Movie"
        movie_schema = NodeSchema(
            id=PropertySchema(name="movie_id", type="STRING", description="Unique identifier for a movie."),
            label="Movie",
            properties=[
                PropertySchema(name="title", type="STRING", description="The title of the movie."),
            ],
            description="A schema representing a movie."
        )

        # Define relationship schema for "ACTED_IN"
        acted_in_schema = RelationshipSchema(
            type="ACTED_IN",
            queryPatterns=[QueryPattern(
                startNode="Person",
                endNode="Movie",
                description="Indicates a person acted in a movie."
            )],
            properties=[
                PropertySchema(name="role", type="STRING", description="The role played by the person in the movie."),
            ],
            description="A schema representing the ACTED_IN relationship."
        )

        # Define the schema in the GraphRAG instance
        self.graph_rag.schema.define(GraphSchema(
            nodes=[person_schema, movie_schema],
            relationships=[acted_in_schema]
        ))

        # Use `merge_nodes` to add Person and Movie nodes
        person_records = [
            {"person_id": 1, "name": "Alice"},
            {"person_id": 2, "name": "Bob"}
        ]
        movie_records = [
            {"movie_id": "M101", "title": "Inception"},
            {"movie_id": "M102", "title": "The Matrix"}
        ]
        self.graph_rag.data.merge_nodes(label="Person", records=person_records)
        self.graph_rag.data.merge_nodes(label="Movie", records=movie_records)

        # Use `merge_relationships` to add ACTED_IN relationships
        relationship_records = [
            {"start_node_id": 1, "end_node_id": "M101", "role": "Protagonist"},
            {"start_node_id": 2, "end_node_id": "M101", "role": "Villain"},
            {"start_node_id": 2, "end_node_id": "M102", "role": "Hacker"}
        ]
        self.graph_rag.data.merge_relationships(
            rel_type="ACTED_IN",
            start_node_label="Person",
            end_node_label="Movie",
            records=relationship_records
        )

        # Verify that relationships were added
        def verify_relationship_count(tx, rel_type):
            result = tx.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            relationship_count = session.execute_read(verify_relationship_count, "ACTED_IN")

        self.assertEqual(relationship_count, 3, f"Expected 3 ACTED_IN relationships, but found {relationship_count}.")
