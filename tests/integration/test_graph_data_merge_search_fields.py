import os
import unittest

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

from graph_nd.graphrag.graph_data import NodeData, RelationshipData, GraphData
from graph_nd.graphrag.graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, SearchFieldSchema


class TestGraphDataMergeWithSearchFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the database connection and embedding model for the test.
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

        # Set up embedding model (e.g., OpenAI)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable for integration tests.")
        cls.embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the database by deleting all nodes and relationships after tests, then close connections.
        """
        with cls.db_client.session() as session:
            # Run the cleanup Cypher query to delete everything
            session.run("MATCH (n) DETACH DELETE n")
            # Drop the indexes
            session.run("DROP INDEX vector_person_bio_text_embedding IF EXISTS")
            session.run("DROP INDEX fulltext_movie_title IF EXISTS")

        cls.db_client.close()

    def test_graph_data_with_text_fields(self):
        """
        Integration test for adding text embedding and full-text index fields.
        """

        # Define the schemas
        # NodeSchema for Person with bio (embedding field)
        person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="person id"),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="person name"),
                PropertySchema(name="age", type="INTEGER", description="person age in years"),
                PropertySchema(name="bio", type="STRING", description="person's biography")
            ],
            searchFields=[
                SearchFieldSchema(name="bio_text_embedding", type="TEXT_EMBEDDING", calculatedFrom="bio")
            ],
            description="A Person with text embedding on bio"
        )

        # NodeSchema for Movie with fulltext index on title
        movie_schema = NodeSchema(
            id=PropertySchema(name="movie_id", type="STRING", description="movie id"),
            label="Movie",
            properties=[
                PropertySchema(name="title", type="STRING", description="The movie title"),
                PropertySchema(name="release_year", type="INTEGER", description="The movie release year"),
            ],
            searchFields=[
                SearchFieldSchema(
                    name="title",
                    type="FULLTEXT",
                    calculatedFrom="title",  # Define fulltext search on "title"
                )
            ],
            description="A Movie with fulltext index on the title"
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
                {"person_id": 1, "name": "Alice", "age": 30, "bio": "Alice is a computer scientist."},
                {"person_id": 2, "name": "Bob", "age": 25, "bio": "Bob is an AI developer and a musician."}
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
                {"start_node_id": 1, "end_node_id": "M101", "role": "Protagonist"},
                {"start_node_id": 2, "end_node_id": "M101", "role": "Villain"},
                {"start_node_id": 2, "end_node_id": "M102", "role": "Hacker"},
            ]
        )

        # Combine into GraphData instance
        graph_data = GraphData(
            nodeDatas=[person_data, movie_data],
            relationshipDatas=[acted_in_data]
        )

        # Load the graph data into the database with embeddings and search indexes
        graph_data.merge(self.db_client, embedding_model=self.embedding_model)

        # Verify the nodes exist in the database
        def verify_node_count(tx, label):
            result = tx.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            person_count = session.execute_read(verify_node_count, "Person")
            movie_count = session.execute_read(verify_node_count, "Movie")

        self.assertEqual(person_count, 2, f"Expected 2 Person nodes, found {person_count}")
        self.assertEqual(movie_count, 2, f"Expected 2 Movie nodes, found {movie_count}")

        # Verify text embeddings on the right nodes
        def verify_embedding_exists(tx, label, embedding_property):
            result = tx.run(
                f"""
                MATCH (n:{label}) 
                WHERE n.{embedding_property} IS NULL
                RETURN COUNT(n) AS missing_count
                """
            )
            return result.single()["missing_count"]

        with self.db_client.session() as session:
            missing_bio_embeddings = session.execute_read(
                verify_embedding_exists,
                label="Person",
                embedding_property="bio_text_embedding"
            )

        self.assertEqual(
            missing_bio_embeddings,
            0,
            f"Expected all Person nodes to have 'bio_text_embedding', but {missing_bio_embeddings} nodes are missing it."
        )

        # Check if the vector index was created and is queryable
        def verify_vector_index_query(tx, index_name, search_prompt, top_k):
            """
            Verify that the vector index is queryable by running a test query against the specified index.
            """
            query_vector = self.embedding_model.embed_query(search_prompt)
            result = tx.run(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', $topK, $queryVector) YIELD node, score
                RETURN count(node) AS match_count
                """,
                topK=top_k,
                queryVector=query_vector
            )
            return result.single()["match_count"]

        with self.db_client.session() as session:
            # Vector index validation using a search prompt
            search_prompt = "computer scientist"
            vector_index_query_count = session.execute_read(
                verify_vector_index_query,
                index_name="vector_person_bio_text_embedding",  # Name of the vector index
                search_prompt=search_prompt,
                top_k=5,  # Retrieve the top 5 matches
            )

            self.assertGreater(
                vector_index_query_count,
                0,
                f"Vector index query failed to return any matches for the prompt '{search_prompt}'."
            )

        # Check if the fulltext index was created and is queryable
        def verify_fulltext_search(tx, search_query):
            result = tx.run(
                f"CALL db.index.fulltext.queryNodes('fulltext_movie_title', '{search_query}') YIELD node "
                "RETURN count(node) AS count"
            )
            return result.single()["count"]

        with self.db_client.session() as session:
            matched_movie_count = session.execute_read(verify_fulltext_search, "The Matrix")
            unmatched_movie_count = session.execute_read(verify_fulltext_search, "Unknown Title")

        self.assertEqual(matched_movie_count, 1, "Fulltext search failed to match a valid Movie title.")
        self.assertEqual(unmatched_movie_count, 0, "Fulltext search incorrectly matched an invalid query.")

        # Verify that relationships exist in the database
        def verify_relationship_count(tx, rel_type):
            result = tx.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
            return result.single()["count"]

        with self.db_client.session() as session:
            relationship_count = session.execute_read(verify_relationship_count, "ACTED_IN")

        self.assertEqual(relationship_count, 3, f"Expected 3 ACTED_IN relationships, found {relationship_count}")
