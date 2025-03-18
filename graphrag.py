import json
import os
from pprint import pprint
from typing import Dict, List

from tqdm import tqdm

from graph_data import NodeData, RelationshipData, GraphData
from graph_schema import GraphSchema, NodeSchema
from graph_records import SubGraph
from table_mapping import TableTypeEnum, TableType, NodeTableMapping, RelTableMapping
from prompt_templates import SCHEMA_FROM_DESC_TEMPLATE, SCHEMA_FROM_SAMPLE_TEMPLATE, SCHEMA_FROM_DICT_TEMPLATE, \
    TABLE_TYPE_TEMPLATE, NODE_MAPPING_TEMPLATE, RELATIONSHIPS_MAPPING_TEMPLATE, TEXT_EXTRACTION_TEMPLATE
from utils import read_csv_preview, read_csv, load_pdf


class GraphRAG:
    def __init__(self, db_client, llm=None, embedding_model=None):
        """
        Initializes the GraphRAG instance.

        Args:
            db_client: The database client for managing the knowledge graph
                       (Assumed to be a Neo4j driver in this code example.)
            llm: The language model for handling inference, queries and response completions.
            embedding_model: text embedding model to use for data
        """
        self.db_client = db_client
        self.llm = llm

        # Initialize Schema and Data components
        self.schema = self.Schema(self.db_client, llm)
        self.data = self.Data(self, self.db_client, llm, embedding_model)

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

        def __init__(self, graphrag, db_client, llm, embedding_model=None):
            """
            Initializes the Data class.

            Args:
                graphrag: A reference to the outer `GraphRAG` instance.
                db_client: The database client for managing the knowledge graph.
            """
            self.embedding_model = None
            self.llm_rels_table_mapping = None
            self.llm_node_table_mapping = None
            self.llm_table_type = None
            self.llm_text_extractor = None
            self.graphrag = graphrag  # Reference to the outer GraphRAG instance
            self.db_client = db_client
            self.set_llms(llm)
            self.set_embedding_model(embedding_model)

        def set_embedding_model(self, embedding_model):
            self.embedding_model = embedding_model if embedding_model else None

        def set_llms(self, llm):
            self.llm_table_type = llm.with_structured_output(TableType, method="function_calling") if llm else None
            self.llm_node_table_mapping = llm.with_structured_output(NodeTableMapping,
                                                                     method="function_calling") if llm else None
            self.llm_rels_table_mapping = llm.with_structured_output(RelTableMapping,
                                                                     method="function_calling") if llm else None
            self.llm_text_extractor = llm.with_structured_output(SubGraph,
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
            node_data.merge(self.db_client, embedding_model=self.embedding_model)

        def merge_relationships(self, rel_type:str, start_node_label:str, end_node_label: str, records: List[Dict]):
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

        def merge_node_table(self, table_records:List[Dict], node_mapping:NodeTableMapping):
                node_records = [node_mapping.convert_to_node_record(rec)['record'] for rec in table_records]
                self.merge_nodes(node_mapping.nodeLabel, node_records)

        def merge_relationships_from_table(self, table_records:List[Dict], rel_mapping:RelTableMapping):
            rel_records = dict()
            node_records = dict()
            dicts_of_triple_records = [rel_mapping.convert_to_triple_records(rec) for rec in table_records]

            ## for each list of triples append to unique rel_records
            for triple_record in dicts_of_triple_records:
                for triple_key, triple_data in triple_record.items():
                    if triple_key not in rel_records: #get relationship metadata
                        rel_records[triple_key] = {"metadata": {'rel_type':triple_data[1]['rel_type'],
                                                            'start_node_label': triple_data[1]['start_node_label'],
                                                            'end_node_label': triple_data[1]['end_node_label']},
                                               "records":[]}
                    rel_records[triple_key]["records"].append(triple_data[1]['record']) #appends the relationship
                    if triple_data[0]['label'] not in node_records:
                        node_records[triple_data[0]['label']] = []
                    node_records[triple_data[0]['label']].append(triple_data[0]['record']) # appends start node
                    if triple_data[2]['label'] not in node_records:
                        node_records[triple_data[2]['label']] = []
                    node_records[triple_data[2]['label']].append(triple_data[2]['record']) # appends end node

            # merge nodes
            for node_label, node_records_list in node_records.items():
                self.merge_nodes(node_label, node_records_list)

            #merge relationships
            for rel_key, rel_data in rel_records.items():
                self.merge_relationships(rel_data['metadata']['rel_type'],
                                        rel_data['metadata']['start_node_label'],
                                        rel_data['metadata']['end_node_label'],
                                        rel_data['records'])

        def merge_node_csv(self, file_path: str):
            table_records = read_csv(file_path)
            table_preview = read_csv_preview(file_path)
            node_mapping = self.get_table_node_mapping(os.path.basename(file_path), table_preview)
            self.merge_node_table(table_records, node_mapping)

        def merge_relationships_csv(self, file_path: str):
            table_records = read_csv(file_path)
            table_preview = read_csv_preview(file_path)
            rel_mapping = self.get_table_relationships_mapping(os.path.basename(file_path), table_preview)
            self.merge_relationships_from_table(table_records, rel_mapping)

        def merge_csv(self, file_path: str):
            table_preview = read_csv_preview(file_path)
            table_type = self.get_table_mapping_type(os.path.basename(file_path), table_preview)
            if table_type == TableTypeEnum.SINGLE_NODE:
                self.merge_node_csv(file_path)
            elif table_type == TableTypeEnum.RELATIONSHIPS:
                self.merge_relationships_csv(file_path)
            else:
                raise ValueError(f"[Data] Unable to determine table type for {file_path}. Got table_type={table_type} instead.")

        def merge_csvs(self, file_paths: List[str]):
            """
            Merges data from CSV files into the knowledge graph.

            Args:
                file_paths (List[str]): The file paths to csvs
            """
            for file_path in file_paths:
                self.merge_csv(file_path)


        def merge_pdf(self, file_path: str, chunk_strategy="BY_PAGE", chunk_size=20):
            """
            Merges data from a pdf file into the knowledge graph.
            """
            texts = load_pdf(file_path=file_path, chunk_strategy=chunk_strategy, chunk_size=chunk_size)
            self._validate_llms()
            for text in tqdm(texts, desc="Extracting entities from PDF"):
                prompt = TEXT_EXTRACTION_TEMPLATE.invoke({'fileName': os.path.basename(file_path),
                                                     'text': text,
                                                     'graphSchema':self.graphrag.schema.schema.prompt_str()})
                #pprint(prompt.text)
                # Use structured LLM for extraction
                extracted_subgraph: SubGraph = self.llm_text_extractor.invoke(prompt)
                graph_data = extracted_subgraph.convert_to_graph_data(self.graphrag.schema.schema)
                # merge nodes
                for node_data in graph_data.nodeDatas:
                    node_data.merge(self.db_client, embedding_model=self.embedding_model)
                # merge relationships
                for rels_data in graph_data.relationshipDatas:
                    rels_data.merge(self.db_client)



            print(f"[Data] Merging data from document: {file_path}")
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

