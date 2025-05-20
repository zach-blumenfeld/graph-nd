import os
import unittest

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from neo4j import GraphDatabase

from graph_nd import GraphRAG
from graph_nd.graphrag.graph_data import NodeData, RelationshipData, GraphData
from graph_nd.graphrag.graph_schema import NodeSchema, PropertySchema, RelationshipSchema, QueryPattern, SearchFieldSchema


class TestGraphSchemaFromExistingGraph(unittest.TestCase):

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

        #instantiate graphrag but don't create schema and load data through it.  We will do that later
        cls.graphrag = GraphRAG(cls.db_client, ChatOpenAI(model="gpt-4o", temperature=0.0), cls.embedding_model)

        #clear graph
        cls.graphrag.data.nuke(skip_confirmation=True)


        # Define the schemas
        # NodeSchema for Person with bio (embedding field)
        cls.person_schema = NodeSchema(
            id=PropertySchema(name="person_id", type="INTEGER", description="person id"),
            label="Person",
            properties=[
                PropertySchema(name="name", type="STRING", description="person name"),
                PropertySchema(name="age", type="INTEGER", description="person age in years"),
                PropertySchema(name="bio", type="STRING", description="person's biography")
            ],
            searchFields=[
                SearchFieldSchema(name="bio_text_embedding", type="TEXT_EMBEDDING", calculatedFrom="bio", indexName="bio_text_embedding")
            ],
            description="A Person with text embedding on bio"
        )

        # NodeSchema for Movie with fulltext index on title
        cls.movie_schema = NodeSchema(
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
        cls.acted_in_schema = RelationshipSchema(
            type="ACTED_IN",
            id=None,  # No unique identifier for relationships in this test
            properties=[
                PropertySchema(name="role", type="STRING", description="The role played by the person in the movie.")
            ],
            queryPatterns=[QueryPattern(startNode="Person", endNode="Movie", description="person acted in movie")],
            description="a person acting in a movie"
        )

        # RelationshipSchema for KNOWS
        cls.knows_schema = RelationshipSchema(
            type="KNOWS",
            id=None,  # No unique identifier for relationships in this test
            properties=[
                PropertySchema(name="relationship", type="STRING", description="The relationship between the two people, e.g., friend, colleague, etc.")
            ],
            queryPatterns=[QueryPattern(startNode="Person", endNode="Person", description="person knows person")],
            description=""
        )

        # Define the node data
        person_data = NodeData(
            node_schema=cls.person_schema,
            records=[
                {"person_id": 1, "name": "Alice", "age": 30, "bio": "Alice is a computer scientist."},
                {"person_id": 2, "name": "Bob", "age": 25, "bio": "Bob is an AI developer and a musician."},
                {"person_id": 3, "name": "Frank", "age": 42, "bio": "Frank is the dude that is good at bowling and stuff."}
            ]
        )

        movie_data = NodeData(
            node_schema=cls.movie_schema,
            records=[
                {"movie_id": "M101", "title": "Inception", "release_year": 2010},
                {"movie_id": "M102", "title": "The Matrix", "release_year": 1999}
            ]
        )

        # Define the relationship data
        acted_in_data = RelationshipData(
            rel_schema=cls.acted_in_schema,
            start_node_schema=cls.person_schema,
            end_node_schema=cls.movie_schema,
            records=[
                {"start_node_id": 1, "end_node_id": "M101", "role": "Protagonist"},
                {"start_node_id": 2, "end_node_id": "M101", "role": "Villain"},
                {"start_node_id": 2, "end_node_id": "M102", "role": "Hacker"},
            ]
        )

        knows_data = RelationshipData(
            rel_schema=cls.knows_schema,
            start_node_schema=cls.person_schema,
            end_node_schema=cls.person_schema,
            records=[
                {"start_node_id": 1, "end_node_id": 3, "relationship": "best friends"},
                {"start_node_id": 2, "end_node_id": 3, "relationship": "bitter enemies"},
            ]
        )

        # Combine into GraphData instance
        graph_data = GraphData(
            nodeDatas=[person_data, movie_data],
            relationshipDatas=[acted_in_data, knows_data]
        )

        # Load the graph data into the database with embeddings and search indexes
        graph_data.merge(cls.db_client, embedding_model=cls.embedding_model)

        # RelationshipSchema for ACTED_IN PARALLEL
        cls.acted_in_schema_parallel = RelationshipSchema(
            type="ACTED_IN",
            id=PropertySchema(name="role", type="STRING", description="The role played by the person in the movie."),
            properties=[],
            queryPatterns=[QueryPattern(startNode="Person", endNode="Movie", description="person acted in movie")],
            description="a person acting in a movie"
        )

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the database by deleting all nodes and relationships after tests, then close connections.
        """
        with cls.db_client.session() as session:
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

                # drop constraints
                session.run("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")
            cls.db_client.close()

    def test_graph_schema_from_existing(self):

        self.graphrag.schema.from_existing_graph()
        self._compare_graph_schemas(True)

    def test_graph_schema_from_existing_with_text_emb_index(self):

        self.graphrag.schema.from_existing_graph(text_embed_index_map={"bio_text_embedding":"bio"})
        self._compare_graph_schemas(False)

    def test_graph_schema_from_existing_throw_error_with_bad_text_emb_ind_name(self):
        """
        Test that an error is thrown when a bad text embedding index mapping is provided.
        The error should be thrown because 'typo_index' is mapped to 'bio' but no such index exists.
        """
        bad_ind_name = "typo_index"
        # We expect this to throw a ValueError due to invalid text_embed_index_map
        with self.assertRaises(ValueError) as context:
            self.graphrag.schema.from_existing_graph(
                text_embed_index_map={"bio_text_embedding": "bio", bad_ind_name: "bio"}
            )
        self.assertIn(f"The Vector index `{bad_ind_name}` does not exist", str(context.exception))

        bad_ind_prop_name = "typo_prop"
        # We expect this to throw a ValueError due to invalid text_embed_index_map
        with self.assertRaises(ValueError) as context:
            self.graphrag.schema.from_existing_graph(
                text_embed_index_map={"bio_text_embedding": bad_ind_prop_name}
            )
        self.assertIn(f"The text embedding field for `{{bio_text_embedding:{bad_ind_prop_name}}}` was never found", str(context.exception))

    def test_graph_schema_from_existing_parallel_rel(self):
        self.graphrag.schema.from_existing_graph(parallel_rel_ids={"ACTED_IN": "role"})
        self._compare_graph_schemas(True, True)

    def test_graph_schema_from_existing_bad_parallel_rel(self):
        bad_rel_type = "ACTED_IN_TYPO"
        with self.assertRaises(ValueError) as context:
            self.graphrag.schema.from_existing_graph(parallel_rel_ids={bad_rel_type: "role"})
        self.assertIn(f"The provided relationship type `{bad_rel_type}` from parallel_rel_ids doesn't exist",
                      str(context.exception))
        bad_rel_id = "role_typo"
        with self.assertRaises(ValueError) as context:
            self.graphrag.schema.from_existing_graph(parallel_rel_ids={"ACTED_IN": bad_rel_id})
        self.assertIn(f"The provided parallel id property `{bad_rel_id}` for relationship type `ACTED_IN` is missing",
                      str(context.exception))


    def _compare_graph_schemas(self, skip_text_embeddings=False, use_parallel_acted_in=False):
        # Get reference schemas
        expected_node_schemas = {schema.label: schema for schema in [self.person_schema, self.movie_schema]}

        if use_parallel_acted_in:
            expected_rel_schemas = {schema.type: schema for schema in [self.acted_in_schema_parallel, self.knows_schema]}
        else:
            expected_rel_schemas = {schema.type: schema for schema in [self.acted_in_schema, self.knows_schema]}

        # Get actual schemas from GraphRAG
        actual_node_schemas = {schema.label: schema for schema in self.graphrag.schema.schema.nodes}
        actual_rel_schemas = {schema.type: schema for schema in self.graphrag.schema.schema.relationships}

        # PART 1: Validate Node Schemas
        # Check that we have the expected number of node schemas
        self.assertEqual(len(actual_node_schemas), len(expected_node_schemas),
                         f"Expected {len(expected_node_schemas)} node schemas, got {len(actual_node_schemas)}")

        # Check that all expected node schemas exist
        for label, expected_schema in expected_node_schemas.items():
            self.assertIn(label, actual_node_schemas, f"Expected node schema '{label}' not found")

            actual_schema = actual_node_schemas[label]

            # Compare nodeId
            self._compare_property_schemas(
                [actual_schema.id],
                [expected_schema.id],
                f"Node schema '{label}' nodeId"
            )

            # Compare properties
            if actual_schema.properties or expected_schema.properties:
                self._compare_property_schemas(
                    actual_schema.properties,
                    expected_schema.properties,
                    f"Node schema '{label}'"
                )

            # Compare search fields
            self._compare_search_field_schemas(
                actual_schema.searchFields,
                expected_schema.searchFields,
                f"Node schema '{label}'",
                skip_text_embeddings=skip_text_embeddings
            )
        # PART 2: Validate Relationship Schemas
        # Check that we have the expected number of relationship schemas
        self.assertEqual(len(actual_rel_schemas), len(expected_rel_schemas),
                         f"Expected {len(expected_rel_schemas)} relationship schemas, got {len(actual_rel_schemas)}")

        # Check that all expected relationship schemas exist
        for rel_type, expected_schema in expected_rel_schemas.items():
            self.assertIn(rel_type, actual_rel_schemas, f"Expected relationship schema '{rel_type}' not found")

            actual_schema = actual_rel_schemas[rel_type]

            # Compare relId
            if actual_schema.id or expected_schema.id:
                self._compare_property_schemas(
                    [actual_schema.id],
                    [expected_schema.id],
                    f"Node schema '{rel_type}' nodeId"
                )

            # Compare properties
            if actual_schema.properties or expected_schema.properties:
                self._compare_property_schemas(
                    actual_schema.properties,
                    expected_schema.properties,
                    f"Relationship schema '{rel_type}'"
                )

            # Compare query patterns
            self._compare_query_patterns(
                actual_schema.queryPatterns,
                expected_schema.queryPatterns,
                f"Relationship schema '{rel_type}'"
            )

    def _compare_property_schemas(self, actual_props, expected_props, context=""):
        """Helper method to compare property schemas."""
        # Convert to dictionaries for easier comparison
        actual_props_dict = {prop.name: prop for prop in actual_props}
        expected_props_dict = {prop.name: prop for prop in expected_props}

        # Check counts
        self.assertEqual(len(actual_props), len(expected_props),
                         f"{context} should have {len(expected_props)} properties, got {len(actual_props)}")

        # Check all expected properties exist with correct types
        for name, expected_prop in expected_props_dict.items():
            self.assertIn(name, actual_props_dict, f"Property '{name}' missing from {context}")

            actual_prop = actual_props_dict[name]
            self.assertEqual(actual_prop.type, expected_prop.type,
                             f"Property type mismatch for {context}.{name}: {actual_prop.type} != {expected_prop.type}")


    def _compare_search_field_schemas(self, actual_fields, expected_fields, context="", skip_text_embeddings=False):
        """Helper method to compare search field schemas."""
        # Convert to dictionaries for easier comparison
        actual_fields_dict = {field.name: field for field in actual_fields}
        expected_fields_dict =( {field.name: field for field in expected_fields} if not skip_text_embeddings
                                else {field.name: field for field in expected_fields if field.type != "TEXT_EMBEDDING"})

        # Check counts
        self.assertEqual(len(actual_fields_dict), len(expected_fields_dict),
                         f"{context} should have {len(expected_fields)} search fields, got {len(actual_fields)}")

        # Check all expected search fields exist with correct attributes
        for name, expected_field in expected_fields_dict.items():
            self.assertIn(name, actual_fields_dict, f"Search field '{name}' missing from {context}")

            actual_field = actual_fields_dict[name]
            self.assertEqual(actual_field.type, expected_field.type,
                             f"Search field type mismatch for {context}.{name}")

            # Check calculatedFrom if it exists
            if hasattr(expected_field, 'calculatedFrom') and expected_field.calculatedFrom:
                self.assertEqual(actual_field.calculatedFrom, expected_field.calculatedFrom,
                                 f"Search field calculatedFrom mismatch for {context}.{name}")


    def _compare_query_patterns(self, actual_patterns, expected_patterns, context=""):
        """Helper method to compare relationship query patterns."""
        # Check counts
        self.assertEqual(len(actual_patterns), len(expected_patterns),
                         f"{context} should have {len(expected_patterns)} query patterns, got {len(actual_patterns)}")

        # Since patterns might not have a natural key, we'll compare them by position
        # This assumes the order is preserved, which is reasonable for test scenarios
        for i, (actual_pattern, expected_pattern) in enumerate(zip(actual_patterns, expected_patterns)):
            self.assertEqual(actual_pattern.startNode, expected_pattern.startNode,
                             f"{context} query pattern {i} startNode mismatch")
            self.assertEqual(actual_pattern.endNode, expected_pattern.endNode,
                             f"{context} query pattern {i} endNode mismatch")



