import json
from pprint import pprint
from typing import Dict, List

from GraphData import NodeData, RelationshipData
from GraphSchema import GraphSchema, NodeSchema
from table_mapping import TableTypeEnum, TableType, NodeTableMapping, RelTableMapping
from prompt_templates import SCHEMA_FROM_DESC_TEMPLATE, SCHEMA_FROM_SAMPLE_TEMPLATE, SCHEMA_FROM_DICT_TEMPLATE, \
    TABLE_TYPE_TEMPLATE, NODE_MAPPING_TEMPLATE, RELATIONSHIPS_MAPPING_TEMPLATE


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
        self.data = self.Data(self, self.db_client, llm)

    def set_llm(self, llm):
        """
        Sets or updates the language model (LLM) for GraphRAG and Schema.

        Args:
            llm: The language model (LLM) instance to use.
        """
        self.llm = llm
        self.schema.set_llm(llm)
        self.data.set_llms(llm)

    class Schema:
        """
        Encapsulates the knowledge graph schema.
        """

        def __init__(self, db_client, llm):
            self.schema = None
            self.db_client = db_client
            self.llm = llm.with_structured_output(GraphSchema, method="function_calling") if llm else None

        def set_llm(self, llm):
            """
            Sets or updates the LLM in the Schema and ensures proper configuration.

            Args:
                llm: The new LLM instance to set.
            """
            self.llm = llm.with_structured_output(GraphSchema, method="function_calling") if llm else None

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
            self.schema = self.llm.invoke(prompt)
            print(f"Generated schema:\n {self.schema}")
            return self.schema

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
            self.schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.schema}")
            return self.schema

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
            self.schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.schema}")
            return self.schema


        def define(self, graph_schema: GraphSchema):
            """
            sets the schema exactly/explicitly using GraphSchema

            Args:
                graph_schema (GraphSchema): The exact schema to use.
            """
            self.schema = graph_schema
            print("[Schema] Defining schema...")
            print(f"[Schema] Schema defined as:\n {graph_schema}")

        def export(self, file_path):
            """
            Exports the current schema to a JSON file.

            Args:
                file_path (str): The path to the file where the schema will be saved.
            """
            if self.schema is None:
                raise ValueError("[Schema] No schema defined to export.")

            try:
                # Convert the schema to a dictionary and write it to a JSON file
                with open(file_path, 'w') as file:
                    json.dump(self.schema.model_dump(), file, indent=4)  # Assuming GraphSchema supports `to_dict()`
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
                self.schema = GraphSchema.model_validate(schema_dict)
                print(f"[Schema] Schema successfully loaded from {file_path}")
            except Exception as e:
                print(f"[Schema] Error loading schema from {file_path}: {e}")
                raise

    class Data:
        """
        Data management for the knowledge graph.
        """

        def __init__(self, graph_rag, db_client, llm):
            """
            Initializes the Data class.

            Args:
                graph_rag: A reference to the outer `GraphRAG` instance.
                db_client: The database client for managing the knowledge graph.
            """
            self.llm_rels_table_mapping = None
            self.llm_node_table_mapping = None
            self.llm_table_type = None
            self.graphrag = graph_rag  # Reference to the outer GraphRAG instance
            self.db_client = db_client
            self.set_llms(llm)

        def set_llms(self, llm):
            self.llm_table_type = llm.with_structured_output(TableType, method="function_calling") if llm else None
            self.llm_node_table_mapping = llm.with_structured_output(NodeTableMapping,
                                                                     method="function_calling") if llm else None
            self.llm_rels_table_mapping = llm.with_structured_output(RelTableMapping,
                                                                     method="function_calling") if llm else None

        def _validate_llms(self):
            if any(attr is None for attr in [self.llm_table_type, self.llm_node_table_mapping, self.llm_rels_table_mapping]):
                raise ValueError("[Data] LLM is not set. Please set the LLM before calling this method.")

        def merge_nodes(self, label:str, records: List[Dict]):
            """
            Merges nodes into the database using the provided label and record data.

            Parameters:
                label (str): The label of the node type to merge (e.g., "Person", "Movie").
                             The label should match a defined node in the graph schema.
                records List[Dict]: A list of dictionaries representing the data for each node to be merged.
                                Each record MUST include the `id` field as defined in the node schema, along with
                                any other optional properties expected by the schema.

            Example:
                label = "Person"
                records = [
                    {"person_id": 1, "name": "Alice", "age": 30},
                    {"person_id": 2, "name": "Bob", "age": 25}
                ]
                Expected Behavior:
                    - Creates or updates nodes labeled "Person" using the records

            Raises:
                ValueError: If the label is not found in the graph schema
            """

            node_schema = self.graphrag.schema.schema.get_node_schema_by_label(label)
            node_data = NodeData(node_schema=node_schema, records=records)
            node_data.merge(self.db_client)

        def merge_relationships(self, rel_type:str, start_node_label:str, end_node_label: str, records: Dict):
            """
            Merges relationships into the database using the provided relationship type, start node label,
            end node label, and record data.

            Parameters:
                rel_type (str): The type of the relationship (e.g., "ACTED_IN", "FRIENDS_WITH").
                                The type should match a defined relationship in the graph schema.
                start_node_label (str): The label of the starting node in the relationship (e.g., "Person").
                                        This label should match a defined node schema.
                end_node_label (str): The label of the ending node in the relationship (e.g., "Movie").
                                      This label should match a defined node schema.
                records (Dict): A dictionary (or list of dictionaries) representing the data for each relationship to be merged.

            Required Fields in `records`:
                - `start_node_id`: The unique identifier of the starting node.
                - `end_node_id`: The unique identifier of the ending node.

            Example:
                rel_type = "ACTED_IN"
                start_node_label = "Person"
                end_node_label = "Movie"
                records = [
                    {"start_node_id": 1, "end_node_id": "M101", "role": "Protagonist"},
                    {"start_node_id": 2, "end_node_id": "M102", "role": "Hacker"}
                ]
                Expected Behavior:
                    - Creates or updates "ACTED_IN" relationships between the "Person" and "Movie" nodes.

            Raises:
                ValueError: If the relationship type, start node label, or end node label is not found in the schema,
                            or if required fields in `records` are missing.
            """

            start_node_schema = self.graphrag.schema.schema.get_node_schema_by_label(start_node_label)
            end_node_schema = self.graphrag.schema.schema.get_node_schema_by_label(end_node_label)
            relationship_schema = self.graphrag.schema.schema.get_relationship_schema(rel_type, start_node_label, end_node_label)
            relationship_data = RelationshipData(rel_schema=relationship_schema,
                                                 start_node_schema=start_node_schema,
                                                 end_node_schema=end_node_schema,
                                                 records=records)
            relationship_data.merge(self.db_client)

        def get_table_mapping_type(self, table_name:str, table_preview: str) -> TableTypeEnum:
            self._validate_llms()
            print(f"[Data] Inferring Table Type of {table_name}")
            prompt = TABLE_TYPE_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            pprint(prompt.text)
            # Use structured LLM for schema inference
            table_type:TableType = self.llm_table_type.invoke(prompt)
            print(f"[Data] Inferred Table Type: {table_type.type}")
            return table_type.type

        def get_table_node_mapping(self, table_name:str, table_preview: str) -> NodeTableMapping:
            self._validate_llms()
            print(f"[Data] Creating node mapping for {table_name}")
            prompt = NODE_MAPPING_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            pprint(prompt.text)
            # Use structured LLM for schema inference
            node_mapping:NodeTableMapping = self.llm_node_table_mapping.invoke(prompt)
            return node_mapping

        def get_table_relationships_mapping(self, table_name:str, table_preview: str) -> RelTableMapping:
            self._validate_llms()
            print(f"[Data] Creating relationships mapping for {table_name}")
            prompt = RELATIONSHIPS_MAPPING_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            pprint(prompt.text)
            # Use structured LLM for schema inference
            rels_mapping:RelTableMapping = self.llm_rels_table_mapping.invoke(prompt)
            return rels_mapping

        def merge_csvs(self, csv_paths: List[str]):
            """
            Merges data from CSV files into the knowledge graph.

            Args:
                csv_paths (List[str]): The file paths to csvs
            """



            # print(f"[Data] Merging data from CSV: {csv_path}")
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

