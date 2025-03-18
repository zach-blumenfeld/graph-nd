import os
import unittest

from dotenv import load_dotenv
from neo4j import GraphDatabase

from graph_data import NodeData, RelationshipData, GraphData
from graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern


class TestGraphDataMerge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the database connection for the test.
        """
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

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the database by deleting all nodes and relationships after tests then close connections.
        """
        with cls.db_client.session() as session:
            # Run the cleanup Cypher query to delete everything
            session.run("MATCH (n) DETACH DELETE n")

        cls.db_client.close()

    def test_graph_data_merge(self):
        """
        Integration test for loading a small graph into the database.
        """

        # Define the schemas
        # NodeSchema for Person
        person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="person id"),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="person name"),
                PropertySchema(name="age", type="INTEGER", description="person age in years")
            ],
            description="A Person"
        )

        # NodeSchema for Movie
        movie_schema = NodeSchema(
            id=PropertySchema(name="movie_id", type="STRING", description="movie id"),
            label="Movie",
            properties=[
                PropertySchema(name="title", type="STRING", description="The movie title"),
                PropertySchema(name="release_year", type="INTEGER", description="The movie title"),
            ],
            description="A Feature Length Movie"
        )

        # RelationshipSchema for ACTED_IN
        acted_in_schema = RelationshipSchema(
            type="ACTED_IN",
            id=None,  # No unique identifier for relationships in this test
            properties=[
                PropertySchema(name="role", type="STRING", description="The role played by the person in the movie.")
            ],
            queryPatterns=[QueryPattern(startNode="Person", endNode="Movie", description="person acted in movie")],
            description="a person acting in a movie"
        )

        # Define the node data
        person_data = NodeData(
            node_schema=person_schema,
            records=[
                {"person_id": 1, "name": "Alice", "age": 30},
                {"person_id": 2, "name": "Bob", "age": 25}
            ]
        )

        movie_data = NodeData(
            node_schema=movie_schema,
            records=[
                {"movie_id": "M101", "title": "Inception", "release_year": 2010},
                {"movie_id": "M102", "title": "The Matrix", "release_year": 1999}
            ]
        )

        # Define the relationship data
        acted_in_data = RelationshipData(
            rel_schema=acted_in_schema,
            start_node_schema=person_schema,
            end_node_schema=movie_schema,
            records=[
                {"start_node_id": 1,"end_node_id": "M101", "role": "Protagonist"},
                {"start_node_id": 2, "end_node_id": "M101", "role": "Villain"},
                {"start_node_id": 2, "end_node_id": "M102", "role": "Hacker"},
            ]
        )

        # Combine into GraphData instance
        graph_data = GraphData(
            nodeDatas=[person_data, movie_data],
            relationshipDatas=[acted_in_data]
        )

        # Load the graph data into the database
        graph_data.merge(self.db_client)

        # Verify that the nodes exist in the database
        def verify_node_count(tx, label):
            result = tx.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            person_count = session.execute_read(verify_node_count, "Person")
            movie_count = session.execute_read(verify_node_count, "Movie")

        self.assertEqual(person_count, 2, f"Expected 2 Person nodes, found {person_count}")
        self.assertEqual(movie_count, 2, f"Expected 2 Movie nodes, found {movie_count}")

        # Verify that relationships exist in the database
        def verify_relationship_count(tx, rel_type):
            result = tx.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            relationship_count = session.execute_read(verify_relationship_count, "ACTED_IN")

        self.assertEqual(relationship_count, 3, f"Expected 2 ACTED_IN relationships, found {relationship_count}")
