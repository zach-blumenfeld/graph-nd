import json
from typing import Dict

from GraphSchema import GraphSchema
from prompt_templates import SCHEMA_FROM_DESC_TEMPLATE, SCHEMA_FROM_SAMPLE_TEMPLATE, SCHEMA_FROM_DICT_TEMPLATE


class GraphRAG:
    def __init__(self, db_client, llm=None):
        """
        Initializes the GraphRAG instance.

        Args:
            db_client: The database client for managing the knowledge graph
                       (Assumed to be a Neo4j driver in this code example.)
            llm: The language model for handling inference, queries and response completions.
        """
        self.db_client = db_client
        self.llm = llm

        # Initialize Schema and Data components
        self.schema = self.Schema(self.db_client, llm)
        self.data = self.Data(self.db_client)

    def set_llm(self, llm):
        """
        Sets or updates the language model (LLM) for GraphRAG and Schema.

        Args:
            llm: The language model (LLM) instance to use.
        """
        self.llm = llm
        self.schema.set_llm(llm)

    class Schema:
        """
        Encapsulates the knowledge graph schema.
        """

        def __init__(self, db_client, llm):
            self.graph_schema = None
            self.db_client = db_client
            self.llm = llm.with_structured_output(GraphSchema) if llm else None

        def set_llm(self, llm):
            """
            Sets or updates the LLM in the Schema and ensures proper configuration.

            Args:
                llm: The new LLM instance to set.
            """
            self.llm = llm.with_structured_output(GraphSchema) if llm else None

        def _validate_llm(self):
            """
            Validates that the LLM is set. Raises an error if not.
            """
            if self.llm is None:
                raise ValueError("[Schema] LLM is not set. Please set the LLM before calling this method.")

        def infer(self, description: str):
            """
            Infers the graph schema based on a description of the data.

            Args:
                description (str): A text description of the data for schema inference.
            """
            self._validate_llm()
            print(f"[Schema] Inferring schema based on: {description}")
            prompt = SCHEMA_FROM_DESC_TEMPLATE.invoke({'context':description})
            # Use structured LLM for schema inference
            self.graph_schema = self.llm.invoke(prompt)
            print(f"Generated schema:\n {self.graph_schema}")
            return self.graph_schema

        def infer_from_sample(self, text: str):
            """
            Infers the graph schema based on a small sample of the data.

            Args:
                text (str): A sample of the data in text form.
            """
            self._validate_llm()
            print(f"[Schema] Inferring schema based on sample data")
            prompt = SCHEMA_FROM_SAMPLE_TEMPLATE.invoke({'context':text})
            # Use structured LLM for schema inference
            self.graph_schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.graph_schema}")
            return self.graph_schema

        def craft_from_dict(self, schema_json: str):
            """
            uses LLM to craft the graph schema based on a JSON-like definition.

            Args:
                schema_json (str): A JSON-like dictionary defining the schema.
            """
            self._validate_llm()
            print(f"[Schema] Crafting schema based on provided dict")
            prompt = SCHEMA_FROM_DICT_TEMPLATE.invoke({'context':schema_json})
            # Use structured LLM for schema inference
            self.graph_schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.graph_schema}")
            return self.graph_schema


        def define(self, graph_schema: GraphSchema):
            """
            sets the schema exactly/explicitly using GraphSchema

            Args:
                graph_schema (GraphSchema): The exact schema to use.
            """
            self.graph_schema = graph_schema
            print("[Schema] Defining schema...")
            print(f"[Schema] Schema defined as:\n {graph_schema}")

        def export(self, file_path):
            """
            Exports the current schema to a JSON file.

            Args:
                file_path (str): The path to the file where the schema will be saved.
            """
            if self.graph_schema is None:
                raise ValueError("[Schema] No schema defined to export.")

            try:
                # Convert the schema to a dictionary and write it to a JSON file
                with open(file_path, 'w') as file:
                    json.dump(self.graph_schema.model_dump(), file, indent=4)  # Assuming GraphSchema supports `to_dict()`
                print(f"[Schema] Schema successfully exported to {file_path}")
            except Exception as e:
                print(f"[Schema] Error exporting schema to {file_path}: {e}")
                raise

        def load(self, file_path):
            """
            Loads schema from a JSON file.

            Args:
                file_path (str): The path to the JSON file containing the schema.
            """
            try:
                # Read the JSON file and reconstruct the Pydantic model
                with open(file_path, 'r') as file:
                    schema_dict = json.load(file)
                self.graph_schema = GraphSchema.model_validate(schema_dict)
                print(f"[Schema] Schema successfully loaded from {file_path}")
            except Exception as e:
                print(f"[Schema] Error loading schema from {file_path}: {e}")
                raise

    class Data:
        """
        Data management for the knowledge graph.
        """

        def __init__(self, db_client):
            self.db_client = db_client



        def merge_csv(self, csv_path: str):
            """
            Merges data from a CSV file into the knowledge graph.

            Args:
                csv_path (str): The file path to a CSV file with headers.
            """
            print(f"[Data] Merging data from CSV: {csv_path}")
            # query = f"""
            # LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
            # MERGE (n:Node {{id: row.id}})
            # SET n += row
            # """
            # with self.db_client.session() as session:
            #     session.run(query)
            #     print("[Data] Data merged from CSV.")

        def merge_doc(self, doc_path: str):
            """
            Merges data from a document file into the knowledge graph.

            Args:
                doc_path (str): Path to the document to be parsed.
            """
            print(f"[Data] Merging data from document: {doc_path}")
            # Placeholder: Implement actual document parsing and merging logic

        def merge_db_tables(self, source_client):
            """
            Merges data from external database tables into the knowledge graph.

            Args:
                source_client: A client to connect to the source database.
            """
            print("[Data] Merging data from external database tables.")
            # Placeholder for database table merging logic

    def agent(self, query: str):
        """
        Answers a question or query about the knowledge graph,
        optionally using an LLM for advanced question-answering tasks.

        Args:
            query (str): The query to be executed.

        Returns:
            str: The result of the query or additional insights via the LLM.
        """
        print(f"[Agent] Handling query: {query}")
        return "Hi!!! - I am useless right now!"

