import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Sequence, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from neo4j import RoutingControl
from tqdm.asyncio import tqdm as tqdm_async


from graph_nd.graphrag.graph_data import NodeData, RelationshipData, GraphData
from graph_nd.graphrag.graph_schema import GraphSchema, SubSchema
from graph_nd.graphrag.graph_records import SubGraph, SubGraphNodes
from graph_nd.graphrag.schema_from_db_utils import create_graph_schema_from_existing_db
from graph_nd.graphrag.source_metadata import SourceType, TransformType, LoadType, prepare_source_metadata
from graph_nd.graphrag.table_mapping import TableTypeEnum, TableType, NodeTableMapping, RelTableMapping
from graph_nd.graphrag.prompt_templates import SCHEMA_FROM_DESC_TEMPLATE, SCHEMA_FROM_SAMPLE_TEMPLATE, \
    SCHEMA_FROM_DICT_TEMPLATE, \
    TABLE_TYPE_TEMPLATE, NODE_MAPPING_TEMPLATE, RELATIONSHIPS_MAPPING_TEMPLATE, TEXT_EXTRACTION_TEMPLATE, \
    QUERY_TEMPLATE, AGG_QUERY_TEMPLATE, INTERNAL_AGENT_SYSTEM_TEMPLATE, TEXT_NODE_EXTRACTION_TEMPLATE, \
    SCHEMA_FROM_USE_CASE_MAPPING_TEMPLATE, AGENT_SYSTEM_TEMPLATE
from graph_nd.graphrag.utils import read_csv_preview, read_csv, load_pdf, remove_key_recursive, run_async_function

class GraphRAG:
    """
    GraphRAG serves as the core end-to-end implementation to
    - create and manage knowledge graph data - mapping from structured & unstructured sources
    - create and mange graph schemas which are core to data validation and LLM/agent prompting
    - generation of tools & agents for GraphRAG workflows

    Attributes:
        db_client: A database client for managing underlying
            graph data (Assumed to be a Neo4j driver).
        llm: A Langchain LLM instance for various data, schema, and agent tasks.
        schema: A Schema instance to manage schema-related operations.
        data: A Data instance for managing knowledge graph data.
        agent_executor: A LangGraph GraphRAG agent.
    """
    def __init__(self, db_client=None, llm:Optional[BaseChatModel]=None, embedding_model:Optional[Embeddings]=None):
        """
        Initializes the GraphRAG instance.

        Args:
            db_client: The database client for managing the knowledge graph
                       (Assumed to be a Neo4j driver)
            llm: LangChain LLM for handling inference, queries and response completions.
            embedding_model: LangChain text embedding model to use for data and semantic search.
        """
        self.agent_executor = None
        self.db_client = db_client
        self.llm = llm

        # Initialize Schema and Data components
        self.schema = self.Schema(self.db_client, llm)
        self.data = self.Data(self, self.db_client, llm, embedding_model)

    def set_llm(self, llm:Optional[BaseChatModel]):
        """
        Sets or updates the language model (LLM) for GraphRAG, Schema, and Data.

        Args:
            llm: The language model (LLM) instance to use.
        """
        self.llm = llm
        self.schema.set_llm(llm)
        self.data.set_llm(llm)

    class Schema:
        """
        Encapsulates the knowledge graph schema which is used to validate data and prompt LLMs and agents for graph interactions.
        """

        def __init__(self, db_client, llm:Optional[BaseChatModel]):
            self.schema:Optional[GraphSchema] = None
            self.db_client = db_client
            self.llm = llm.with_structured_output(GraphSchema, method="function_calling") if llm else None

        def set_llm(self, llm:Optional[BaseChatModel]):
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
            prompt = SCHEMA_FROM_DESC_TEMPLATE.invoke({'context':description})
            # Use structured LLM for schema inference
            self.schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.schema.prompt_str()}")
            return self.schema

        def infer_from_sample(self, text: str):
            """
            Infers the graph schema based on a small sample of the data.

            Args:
                text (str): A sample of the data in text form.
            """
            self._validate_llm()
            prompt = SCHEMA_FROM_SAMPLE_TEMPLATE.invoke({'context':text})
            # Use structured LLM for schema inference
            self.schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.schema.prompt_str()}")
            return self.schema

        def infer_from_use_case(self, use_case: str, data_source_models: str = 'No Details Available'):
            """
            Infers a schema from a given use case and external data sources using a
            structured large language model (LLM). This method generates a schema
            prompt based on the supplied context and invokes the LLM to produce
            a schema.

            Args:
                use_case: A string describing the specific use case scenario from
                    which the schema is to be inferred.
                data_source_models: A string representing external data source schemas
                    which are used to populate the graph.

            Returns:
                An inferred schema object generated by the structured LLM model that
                represents the schema derived from the provided use case and external
                data sources.

            Raises:
                ValidationError: If the LLM is not validated before invoking schema
                    generation.
            """
            self._validate_llm()
            prompt = SCHEMA_FROM_USE_CASE_MAPPING_TEMPLATE.invoke({'useCase':use_case, 'sourceDataModels':data_source_models})
            # Use structured LLM for schema inference
            self.schema = self.llm.invoke(prompt)
            print(f"[Schema] Generated schema:\n {self.schema.prompt_str()}")
            return self.schema

        def craft_from_json(self, schema_json: str, verbose=False):
            """
            Crafts a schema object from JSON input using a large language model (LLM) for
            schema inference. This method validates the LLM, generates a structured prompt
            based on the input JSON, and invokes the LLM to produce the schema.

            Parameters:
                schema_json (str): The JSON string representing the schema from which the
                schema needs to be crafted.

                verbose (bool, optional): A flag to determine whether to print the full
                detailed schema or a brief version. Defaults to False.

            Returns:
                The crafted schema object produced by the LLM.
            """
            self._validate_llm()
            prompt = SCHEMA_FROM_DICT_TEMPLATE.invoke({'context':schema_json})
            # Use structured LLM for schema inference
            self.schema = self.llm.invoke(prompt)
            if verbose:
                print(f"[Schema] Generated schema:\n {self.schema.prompt_str()}")
            else:
                print(f"[Schema] Successfully Crafted schema")
            return self.schema

        def from_json_like_file(self, file_path, verbose=False) -> GraphSchema:
            """
            Reads a JSON-like model from a file and crafts a GraphSchema object.

            This method reads the content of a specified file containing a JSON-like model
            definition, and then processes it to generate a GraphSchema object using the
            craft_from_json method. Optionally, it can provide verbose output during the
            process.

            Args:
                file_path: The path to the file containing the JSON-like model to load.
                verbose: A boolean flag indicating whether verbose output of the resulting schema should be enabled
                . Defaults to False.

            Returns:
                A GraphSchema object generated from the JSON-like model in the file.

            Raises:
                IOError: If there's an issue opening or reading the specified file.
                Any other exception raised by the craft_from_json method in the crafting
                process.
            """

            with open(file_path, 'r') as f:
                json_like_model = f.read()
            return self.craft_from_json(json_like_model, verbose)

        def define(self, graph_schema: GraphSchema):
            """
            sets the schema exactly/explicitly using GraphSchema

            Args:
                graph_schema (GraphSchema): The exact schema to use.
            """
            self.schema = graph_schema
            print(f"[Schema] Schema defined as:\n {self.schema.prompt_str()}")

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
                self.schema.export(file_path)
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

        def from_existing_graph(self,
                                exclude_prefixes=("_", " "),
                                exclude_exact_matches=None,
                                text_embed_index_map: Optional[Dict[str, str]] = None,
                                parallel_rel_ids:Optional[Dict[str,str]]=None,
                                description=None):
            """
            Generates a schema from existing graph data.  U
            se for an existing graph database you are connecting for GraphRAG.

            This function analyzes the existing database to generate node schemas and
            relationship schemas, combining them into a unified graph schema. Users can
            specify customization options such as excluding certain prefixes or specific
            exact matches for attributes, and optionally map text embeddings or parallel
            relationship IDs. A description for the schema can also be provided.

            Args:
                exclude_prefixes: A tuple of strings containing prefixes. Node labels, relationship types, or properties
                    starting with any of these prefixes are excluded, defaults to ("_", " ").
                exclude_exact_matches: An optional set of exact node labels, relationship types, or property names to
                    exclude from the schema, defaults to None if not provided.
                text_embed_index_map: An optional dictionary mapping {text_embedding_index_name: text_property}
                    where text_property is a node property that is used to calculate the embedding. This is required to use
                    text embedding search fields for nodes. If not provided, no text embedding search fields will be included in the schema.
                    Defaults to None.
                parallel_rel_ids (Optional[Dict[str, str]], optional): A dictionary mapping relationship
                    types to their parallel relationship ID property names: `{rel_type: property_name}`. This is only required if the
                    user wishes to ingest more data while maintaining parallel relationships for specific node types
                    (more than one instance of a relationship type existing between the same start and end nodes). Defaults to None.
                description: Optional description of the generated graph schema. Exposed to LLM when accessing the graph through GraqphRAG.

            Returns:
                GraphSchema: The graph schema constructed, comprising node schemas
                and relationship schemas extracted from the existing database.
            """
            self.schema = create_graph_schema_from_existing_db(self.db_client,
                                                               exclude_prefixes,
                                                               exclude_exact_matches,
                                                               text_embed_index_map,
                                                               parallel_rel_ids,
                                                               description)
            print(f"[Schema] Schema generated from existing graph")


        def prompt_str(self):
            """
            Returns the prompt string from the associated graph schema.
            This is created from a custom model_dump where Relationships serialize query patterns in the format:
            (:startNodeLabel)-[:TYPE]->(:endNodeLabel). This makes it easier for LLMs and humans to interpret.

            Returns:
                str: The prompt string extracted from the schema's definition.
            """
            return self.schema.prompt_str()

    class Data:
        """
        Data management for the knowledge graph.
        This class is responsible for all write and admin operations on the knowledge graph.
        All source data is mapped through this class's methods.
        """

        def __init__(self, graphrag, db_client, llm:Optional[BaseChatModel], embedding_model:Optional[Embeddings]=None):
            """
            Initializes the Data class.

            Args:
                graphrag: A reference to the outer `GraphRAG` instance.
                db_client: The database client for managing the knowledge graph.
                llm: The LangChain LLM for data operations.
                embedding_model: The LanChain embedding model for  data operations.
            """
            self.llm_node_text_extractor = None
            self.embedding_model = None
            self.llm_rels_table_mapping = None
            self.llm_node_table_mapping = None
            self.llm_table_type = None
            self.llm_text_extractor = None
            self.graphrag = graphrag  # Reference to the outer GraphRAG instance
            self.db_client = db_client
            self.set_llm(llm)
            self.set_embedding_model(embedding_model)

        def set_embedding_model(self, embedding_model:Optional[Embeddings]):
            self.embedding_model = embedding_model if embedding_model else None

        def set_llm(self, llm:Optional[BaseChatModel]):
            """
            Sets the language model (LLM) for all data operations.

            Parameters
            ----------
            llm : Language model, optional
                The language model instance to be set. The language model should support
                a `with_structured_output` method for structured output handling.
            """
            self.llm_table_type = llm.with_structured_output(TableType, method="function_calling") if llm else None
            self.llm_node_table_mapping = llm.with_structured_output(NodeTableMapping,
                                                                     method="function_calling") if llm else None
            self.llm_rels_table_mapping = llm.with_structured_output(RelTableMapping,
                                                                     method="function_calling") if llm else None
            self.llm_text_extractor = llm.with_structured_output(SubGraph,
                                                                 method="function_calling") if llm else None
            self.llm_node_text_extractor = llm.with_structured_output(SubGraphNodes,
                                                                 method="function_calling") if llm else None

        def _validate_llm(self):
            if any(attr is None for attr in [self.llm_table_type,
                                             self.llm_node_table_mapping,
                                             self.llm_rels_table_mapping,
                                             self.llm_text_extractor,
                                             self.llm_node_text_extractor,]):
                raise ValueError("[Data] LLM is not set. Please set the LLM before calling this method.")

        def merge_nodes(self, label:str, records: List[Dict], source_metadata: Union[bool, Dict[str, Any]] = True):
            """
            Merges node data into the graph using the provided label and record data.

            Parameters:
                label (str): The label of the node type to merge (e.g., "Person", "Movie").
                    The label should match a defined node in the graph schema.
                records List[Dict]: A list of dictionaries representing the data for each node to be merged.
                    Each record MUST include the `id` field as defined in the node schema, along with
                    any other optional properties expected by the schema.
                source_metadata : Union[bool, Dict[str, Any]], optional
                    Metadata for the source being merged.
                    - If set to `True`, default source metadata is prepared and added to a __Source__ node in the graph.
                    A __source_id property is added and/or appended to each node which maps to the id property of __Source__ node
                    - If `False`, no source metadata is added to the graph.
                    - If a custom dictionary is provided, source metadata is added as in the case of `True` and the dictionary properties override the default ones.
                    Default is True.

            Example:
                label = "Person"
                records = [
                    {"person_id": 1, "name": "Alice", "age": 30},
                    {"person_id": 2, "name": "Bob", "age": 25}
                ]

            Expected Behavior:
                Creates or updates nodes labeled "Person" using the records

            Raises:
                ValueError: If the label is not found in the graph schema

            """


            node_schema = self.graphrag.schema.schema.get_node_schema_by_label(label)
            node_data = NodeData(node_schema=node_schema, records=records)
            node_data.merge(self.db_client, source_metadata, embedding_model=self.embedding_model)

        def merge_relationships(self, rel_type:str, start_node_label:str, end_node_label: str, records: List[Dict],
                                source_metadata: Union[bool, Dict[str, Any]] = True):
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
                source_metadata : Union[bool, Dict[str, Any]], optional
                            Metadata for the source being merged.
                            - If set to `True`, default source metadata is prepared and added to a __Source__ node in the graph.
                            A __source_id property is added and/or appended to each node and relationship which maps to the id property of __Source__ node
                            - If `False`, no source metadata is added to the graph.
                            - If a custom dictionary is provided, source metadata is added as in the case of `True` and the dictionary properties override the default ones.
                            Default is True.

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
            relationship_data.merge(self.db_client, source_metadata)

        def get_table_mapping_type(self, table_name:str, table_preview: str) -> TableTypeEnum:
            """
            Determines the type of the provided table based on its name, a preview of its data,
            and the schema of the graph. This function uses a language model prompt
            to infer the table type in a structured manner.

            Args:
                table_name (str): Name of the table to infer the type for.
                table_preview (str): A preview or sample content of the table.

            Returns:
                TableTypeEnum: The inferred type of the table.
            """
            self._validate_llm()
            #print(f"[Data] Inferring Table Type of {table_name}")
            prompt = TABLE_TYPE_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            #pprint(prompt.text)
            # Use structured LLM for schema inference
            table_type:TableType = self.llm_table_type.invoke(prompt)
            #print(f"[Data] Inferred Table Type: {table_type.type}")
            return table_type.type

        def get_table_node_mapping(self, table_name:str, table_preview: str) -> NodeTableMapping:
            """
            Generate a table-to-node mapping for the specified table.

            This function generates a mapping between a table and node by using
            an LLM with structured output. It validates the LLM instance,
            invokes a prompt with the given table details, and utilizes the structured
            LLM for schema inference to produce the mapping.

            Parameters:
                table_name: str
                    The name of the table for which the node mapping is to be generated.
                table_preview: str
                    A string representation or preview of the table data.

            Returns:
                NodeTableMapping
                    The mapping object that represents the relationship between the table
                    and its corresponding nodes.

            """
            self._validate_llm()
            #print(f"[Data] Creating node mapping for {table_name}")
            prompt = NODE_MAPPING_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            #pprint(prompt.text)
            # Use structured LLM for schema inference
            node_mapping:NodeTableMapping = self.llm_node_table_mapping.invoke(prompt)
            return node_mapping

        def get_table_relationships_mapping(self, table_name:str, table_preview: str) -> RelTableMapping:
            """
            Generates a table-to-relationships mapping using an LLM with structured outputs.

            This method accepts a table name and its preview to generate a relationship
            mapping by leveraging a large language model (LLM). It invokes a
            specific prompt template for generating the relationships mapping based on the
            table's name, preview, and schema information. The LLM output is then processed
            to create a `RelTableMapping` object which captures the relationships and start and end nodes in the
            provided table.

            Args:
                table_name: str
                    The name of the table for which the relationships mapping is to be created.
                table_preview: str
                    A preview string representation of the table, providing sample data or
                    structure information.

            Returns:
                RelTableMapping
                    A table-to-relationships mapping.
            """
            self._validate_llm()
            #print(f"[Data] Creating relationships mapping for {table_name}")
            prompt = RELATIONSHIPS_MAPPING_TEMPLATE.invoke({'tableName': table_name,
                                                 'tablePreview': table_preview,
                                                 'graphSchema':self.graphrag.schema.schema.prompt_str()})
            #pprint(prompt.text)
            # Use structured LLM for schema inference
            rels_mapping:RelTableMapping = self.llm_rels_table_mapping.invoke(prompt)
            return rels_mapping

        def merge_node_table(self, table_records:List[Dict], node_mapping:NodeTableMapping, source_metadata: Union[bool, Dict[str, Any]] = True):
            """
               Merges data from the provided table records into knowledge graph nodes based on the
               specified node mapping and source metadata.

               Args:
                   table_records (List[Dict]): The records of the table from CSV or other sources to
                       be merged into the knowledge graph.
                   node_mapping (NodeTableMapping): The mapping between the table and node schema
                   source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                       source information of the incoming data. If True, source metadata will
                       be inferred; if False, no source metadata is used; or a dictionary
                       can be passed to define custom metadata. Defaults to True.
            """
            node_records = [node_mapping.convert_to_node_record(rec)['record'] for rec in table_records]
            default_source_metadata = {
                "id": f"merge_nodes_from_table_{os.path.basename(node_mapping.tableName)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "sourceType": SourceType.STRUCTURED_TABLE.value,
                "transformType": TransformType.TABLE_MAPPING_TO_NODE.value,
                "loadType": LoadType.MERGE_NODES.value,
                "name": os.path.basename(node_mapping.tableName),
                "description": node_mapping.tableDescription,
                "file": node_mapping.tableName,
            }

            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)

            self.merge_nodes(node_mapping.nodeLabel, node_records, source_metadata)

        def merge_relationships_from_table(self, table_records:List[Dict], rel_mapping:RelTableMapping, source_metadata: Union[bool, Dict[str, Any]] = True):
            """
            Merges data from the provided table records into knowledge graph relationships and start and end nodes based on the
            specified relationship mapping and source metadata.

            Args:
               table_records (List[Dict]): The records of the table from CSV or other sources to
                   be merged into the knowledge graph.
               rel_mapping (NodeTableMapping): The mapping between the table and relationships and start/end node schemas
               source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.

            """
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

            default_source_metadata = {
                "id": f"merge_nodes_and_rels_from_table_{os.path.basename(rel_mapping.tableName)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "sourceType": SourceType.STRUCTURED_TABLE.value,
                "transformType": TransformType.TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS.value,
                "loadType": LoadType.MERGE_NODES_AND_RELATIONSHIPS.value,
                "name": os.path.basename(rel_mapping.tableName),
                "description": rel_mapping.tableDescription,
                "file": rel_mapping.tableName,
            }
            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)

            # merge nodes
            for node_label, node_records_list in node_records.items():
                self.merge_nodes(node_label, node_records_list, source_metadata)

            #merge relationships
            for rel_key, rel_data in rel_records.items():
                self.merge_relationships(rel_data['metadata']['rel_type'],
                                        rel_data['metadata']['start_node_label'],
                                        rel_data['metadata']['end_node_label'],
                                        rel_data['records'],
                                        source_metadata)

        def merge_node_csv(self, file_path: str, source_metadata: Union[bool, Dict[str, Any]] = True):
            """
            Maps a CSV file to nodes and merges into the knowledge graph.

            This method reads a CSV file to retrieve table records and table preview data, determines the
            node mapping for the table based on the file name and table preview using an LLM workflow, and finally merges the
            table data into the knowledge graph using the node mapping. Optional
            source metadata can be passed or it will be generated with default values.

            Args:
                file_path: str
                    The path to the CSV file to be processed.
               source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
            """
            table_records = read_csv(file_path)
            table_preview = read_csv_preview(file_path)
            node_mapping = self.get_table_node_mapping(os.path.basename(file_path), table_preview)
            default_source_metadata = {
                "id": f"merge_nodes_from_csv_table_{os.path.basename(file_path)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "sourceType": SourceType.STRUCTURED_TABLE_CSV.value,
                "transformType": TransformType.LLM_TABLE_MAPPING_TO_NODE.value,
                "name": os.path.basename(file_path),
                "file": file_path,
            }
            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
            self.merge_node_table(table_records, node_mapping, source_metadata)

        def merge_relationships_csv(self, file_path: str, source_metadata: Union[bool, Dict[str, Any]] = True):
            """
             Maps a CSV file to relationships and their stat/end nodes and merges into the knowledge graph.

            This method reads a CSV file to retrieve table records and table preview data, determines the
            relationships mapping for the table based on the file name and table preview using an LLM workflow, and finally merges the
            table data into the knowledge graph using the relationship mapping. Optional
            source metadata can be passed or it will be generated with default values.

            Args:
                file_path: str
                    The path to the CSV file to be processed.
               source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
            """
            table_records = read_csv(file_path)
            table_preview = read_csv_preview(file_path)
            rel_mapping = self.get_table_relationships_mapping(os.path.basename(file_path), table_preview)
            default_source_metadata = {
                "id": f"merge_nodes_and_rels_from_csv_table_{os.path.basename(file_path)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "sourceType": SourceType.STRUCTURED_TABLE_CSV.value,
                "transformType": TransformType.LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS.value,
                "name": os.path.basename(file_path),
                "file": file_path,
            }
            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
            self.merge_relationships_from_table(table_records, rel_mapping, source_metadata)

        def merge_csv(self, file_path: str, source_metadata: Union[bool, Dict[str, Any]] = True):
            """
            Merges data from a CSV file into the knowledge graph by determining its table type and invoking the appropriate merge method.

            The method identifies the type of data within the CSV file (either a single node or relationships) and
            creates a mapping to convert it to the graph schema using an LLM-powered workflow.

            Args:
                file_path: str
                    The path to the CSV file to be processed.
               source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
            """
            table_preview = read_csv_preview(file_path)
            table_type = self.get_table_mapping_type(os.path.basename(file_path), table_preview)
            print(f"[Data] Merging {os.path.basename(file_path)} as {table_type}.")
            if table_type == TableTypeEnum.SINGLE_NODE:
                self.merge_node_csv(file_path, source_metadata)
            elif table_type == TableTypeEnum.RELATIONSHIPS:
                self.merge_relationships_csv(file_path, source_metadata)
            else:
                raise ValueError(f"[Data] Unable to determine table type for {file_path}. Got table_type={table_type} instead.")

        def merge_csvs(self, file_paths: List[str], source_metadata: Union[bool, Dict[str, Any]] = True):
            """
            Merges data from CSV files into the knowledge graph by determining table types and invoking appropriate merge methods.

            The method identifies the types of data within each CSV file (either a single node or relationships) and
            creates mappings to convert them to the graph schema using an LLM-powered workflow.

            Args:
                file_paths: List[str]
                    The paths to the CSV files to be processed.
               source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
            """
            for file_path in file_paths:
                self.merge_csv(file_path, source_metadata)

        async def _extract_nodes_from_text(self, file_path, text, sub_schema:SubSchema=None) -> GraphData:
            graph_schema:GraphSchema = self.graphrag.schema.schema.subset(sub_schema) if sub_schema else self.graphrag.schema.schema

            prompt = TEXT_NODE_EXTRACTION_TEMPLATE.invoke({'fileName': os.path.basename(file_path),
                                                      'text': text,
                                                      'graphSchema': graph_schema.nodes_only_prompt_str()})
            # pprint(prompt.text)
            # Use structured LLM for extraction
            extracted_nodes: SubGraphNodes = await self.llm_node_text_extractor.ainvoke(prompt)
            graph_data = extracted_nodes.to_subgraph().convert_to_graph_data(self.graphrag.schema.schema)
            return graph_data

        async def _extract_nodes_and_rels_from_text(self, file_path, text, sub_schema:SubSchema=None) -> GraphData:
            graph_schema: GraphSchema = self.graphrag.schema.schema.subset(sub_schema) if sub_schema else self.graphrag.schema.schema
            prompt = TEXT_EXTRACTION_TEMPLATE.invoke({'fileName': os.path.basename(file_path),
                                                      'text': text,
                                                      'graphSchema': graph_schema.prompt_str()})
            # pprint(prompt.text)
            # Use structured LLM for extraction
            extracted_subgraph: SubGraph = await self.llm_text_extractor.ainvoke(prompt)
            graph_data:GraphData = extracted_subgraph.convert_to_graph_data(self.graphrag.schema.schema)
            return graph_data

        async def _extract_from_text_async(self, text, semaphore, source_name: str, nodes_only=True, sub_schema:SubSchema=None) -> GraphData:
            async with semaphore:
                graph_data = await self._extract_nodes_from_text(source_name, text, sub_schema) if nodes_only \
                    else await self._extract_nodes_and_rels_from_text(source_name, text, sub_schema)
                return graph_data

        async def _extract_from_texts_async(self,
                                            texts: List[str],
                                            source_name: str,
                                            nodes_only=True,
                                            max_workers=10,
                                            sub_schema:SubSchema=None) -> GraphData:
            self._validate_llm()
            # Create a semaphore with the desired number of workers
            semaphore = asyncio.Semaphore(max_workers)

            # Create tasks with the semaphore
            tasks = [self._extract_from_text_async(text, semaphore, source_name, nodes_only, sub_schema) for text in texts]

            # Explicitly update progress using `tqdm` as tasks complete
            results:GraphData = GraphData(nodeDatas=[], relationshipDatas=[])
            with tqdm_async(total=len(tasks), desc="Extracting entities from text") as pbar:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    results.nodeDatas.extend(result.nodeDatas)
                    results.relationshipDatas.extend(result.relationshipDatas)
                    pbar.update(1)  # Increment progress bar for each completed task
            print("Consolidating results...")
            results.consolidate()
            return results

        def extract_from_texts(self,
                               texts: List[str],
                               source_name: str,
                               nodes_only=True,
                               max_workers=10,
                               sub_schema:SubSchema=None) -> GraphData:
            """
            Performs entity extraction on a list of text chunks according to the graph schema and outputs the result
            in the form of a GraphData object which contains structured node/relationship records and schem references.

            This method asynchronously processes a list of input texts and extracts relevant
            data to a knowledge graph structure using an LLM-powered workflow.
            Extraction is driven by the GraphSchema.

            Args:
                texts: List[str]
                    A list of input strings (text chunks) from which data will be extracted and structured
                    into a graph representation.
                source_name: str
                    A source identifier or label associated with the input texts. Used for
                    additional context in LLM workflow.
                nodes_only: bool
                    A flag indicating whether to extract only nodes (True) or both nodes and
                    relationships (False) during entity extraction. Defaults to True.
                max_workers: int
                    The maximum number of concurrent workers to use during entity extraction. This
                    parameter affects the level of parallelism when handling input texts.
                    Defaults to 10.
                sub_schema: SubSchema
                    A sub-schema specifying filtering criteria (nodes, patterns, relationships)
                    for the target graphSchema as well as additional description for guiding LLM entity extraction.
                    If not provided, the whole graphSchema is considered. Default is None.

            Returns:
                GraphData
                    A graph representation of the extracted data.
            """
            return run_async_function(self._extract_from_texts_async,
                                      texts,
                                      source_name,
                                      nodes_only,
                                      max_workers,
                                      sub_schema)

        def merge_texts(self,
                        texts: List[str],
                        source_name: str,
                        nodes_only=True,
                        max_workers=10,
                        source_metadata: Union[bool, Dict[str, Any]] = True,
                        sub_schema:SubSchema=None):
            """
            Performs entity extraction on a list of text chunks according to the graph schema, produces
            subgraphs (nodes and relationships), and merges them into the knowledge graph.

            This method asynchronously processes a list of input texts, extracts relevant
            data to a knowledge graph structure using an LLM-powered workflow.
            Extraction is driven by the GraphSchema.

            Args:
                texts: List[str]
                    A list of input strings (text chunks) from which data will be extracted and structured
                    into a graph representation.
                source_name: str
                    A source identifier or label associated with the input texts. Used for
                    additional context in LLM workflow.
                nodes_only: bool
                    A flag indicating whether to extract only nodes (True) or both nodes and
                    relationships (False) during entity extraction. Defaults to True.
                max_workers: int
                    The maximum number of concurrent workers to use during entity extraction. This
                    parameter affects the level of parallelism when handling input texts.
                    Defaults to 10.
                source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
                sub_schema: SubSchema
                    A sub-schema specifying filtering criteria (nodes, patterns, relationships)
                    for the target graphSchema as well as additional description for guiding LLM entity extraction.
                    If not provided, the whole graphSchema is considered. Default is None.
            """
            graph_data:GraphData = self.extract_from_texts(texts, source_name, nodes_only, max_workers, sub_schema)

            default_source_metadata = {
                "sourceType": SourceType.UNSTRUCTURED_TEXT.value,
                "name": os.path.basename(source_name),
                "file": source_name,
            }
            if nodes_only:
                default_source_metadata["id"] = f"merge_nodes_from_text_{os.path.basename(source_name)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                default_source_metadata["transformType"] = TransformType.LLM_TEXT_EXTRACTION_TO_NODES.value
                default_source_metadata["loadType"] = LoadType.MERGE_NODES.value
            else:
                default_source_metadata["id"] = f"merge_nodes_and_rels_from_text_{os.path.basename(source_name)}_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                default_source_metadata["transformType"] = TransformType.LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS.value
                default_source_metadata["loadType"] = LoadType.MERGE_NODES_AND_RELATIONSHIPS.value

            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)

            graph_data.merge(self.db_client, embedding_model=self.embedding_model, source_metadata=source_metadata)

        def merge_pdf(self, file_path: str, chunk_strategy="BY_PAGE", chunk_size=10, nodes_only=True, max_workers=10,
                      source_metadata: Union[bool, Dict[str, Any]] = True, sub_schema:SubSchema=None):
            """

            Merges data from a pdf file into the knowledge graph

            This method
            1. Splits a pdf file into text chunks
            2. Performs parallelized/asynchronous entity extraction on text chunks according to the graph schema using an LLM-powered workflow
            3. Merges extracted subgraphs (nodes and relationships) into knowledge graph

            Args:
                file_path (str): The file path of the PDF document that needs to be processed.
                chunk_strategy (str, optional): The strategy for splitting text into chunks.
                    Currently only supports "BY_PAGE" which splits by pdf page. If you need custom chunking strategies,
                    pre-process your PDF into text chunks and use the `merge_texts` method with your resulting
                    chunks instead. Default is "BY_PAGE".
                chunk_size (int, optional): The size of the chunks for text splitting based on the strategy.
                    Default is 10.
                nodes_only: bool
                    A flag indicating whether to extract only nodes (True) or both nodes and
                    relationships (False) during entity extraction. Defaults to True.
                max_workers: int
                    The maximum number of concurrent workers to use during entity extraction. This
                    parameter affects the level of parallelism when handling input texts.
                    Defaults to 10.
                source_metadata (Union[bool, Dict[str, Any]]): Metadata indicating the
                   source information of the incoming data. If True, default source metadata will
                   be generated; if False, no source metadata is used; or a dictionary
                   can be passed to define custom metadata. Defaults to True.
                sub_schema: SubSchema
                    A sub-schema specifying filtering criteria (nodes, patterns, relationships)
                    for the target graphSchema as well as additional description for guiding LLM entity extraction.
                    If not provided, the whole graphSchema is considered. Default is None.

            Raises:
                ValueError: If the file_path is invalid, empty, or not a supported PDF file.
                RuntimeError: If the LLM validation fails or if any processing error occurs during text
                    extraction or graph merging.
            """
            print(f"[Data] Merging data from document: {file_path}")
            texts = load_pdf(file_path=file_path, chunk_strategy=chunk_strategy, chunk_size=chunk_size)
            default_source_metadata = {
                "sourceType": SourceType.UNSTRUCTURED_TEXT_PDF_FILE.value,
                "name": os.path.basename(file_path),
                "file": file_path,
            }

            source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
            self.merge_texts(texts, file_path, nodes_only, max_workers, source_metadata, sub_schema)

        def nuke(self, delete_chunk_size=10_000, skip_confirmation=False):
            """
            Deletes all nodes, relationships, constraints, and search indexes from
            the database in a batch-wise manner. This method ensures
            efficient cleanup and resets the database to a blank state.

            Args:
                delete_chunk_size : int, optional
                    Number of rows to process per transaction during deletion, by default 10,000.
                skip_confirmation : bool, optional
                    If True, skips the confirmation prompt, by default False.
            """
            if not skip_confirmation:
                confirmation = input("[WARNING] This action will permanently delete all graph data and indexes in the "
                                     "database. Are you sure you want to proceed? (yes/no): ")
                if confirmation.lower() != 'yes':
                    print("[Nuke] Operation aborted by the user.")
                    return

            with self.db_client.session() as session:
                # Delete all nodes and relationships
                session.run(f'''
                MATCH (n)
                CALL (n){{
                  DETACH DELETE n
                }} IN TRANSACTIONS OF {delete_chunk_size} ROWS;
                ''')

                # Drop constraints
                session.run("CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *")

                # Retrieve and drop search indexes
                result = session.run("""
                    SHOW INDEXES YIELD name, type
                    WHERE type IN ["FULLTEXT", "VECTOR"]
                    RETURN name
                """)
                for record in result:
                    index_name = record["name"]
                    session.run(f"DROP INDEX {index_name} IF EXISTS")

    def get_search_configs_prompt(self) -> str:
        search_args_prompt = ''
        for node in self.schema.schema.nodes:
            if node.searchFields:
                if len(node.searchFields):
                    for search_field in node.searchFields:
                        search_type = "SEMANTIC" if search_field.type == "TEXT_EMBEDDING" else search_field.type
                        additional_instruction = f"Additionally: {search_field.description}" if search_field.description and len(search_field.description) > 0 else ''
                        desc = f"Searches {node.label} nodes on the {search_field.calculatedFrom} property using {search_type.lower()} search. {additional_instruction}"
                        search_arg = {
                            "search_type":search_type,
                            "node_label":node.label,
                            "search_prop":search_field.calculatedFrom
                        }
                        search_args_prompt+=str(search_arg) + ": " + desc + "\n"
        return search_args_prompt if len(search_args_prompt) > 0 else 'Apologies no Search Fields Aviable yet.  Advise User and work with other tools.'

    def node_fulltext_search(self,
                              search_query:str,
                              node_label: str,
                              search_prop:str,
                              top_k=10) -> str:

        index_name = self.schema.schema.get_node_search_field(node_label, search_prop, "FULLTEXT").indexName
        return_props = self.schema.schema.get_node_properties(node_label)
        return_statement = ', '.join([f'node.`{p}` AS `{p}`' for p in return_props])
        query = f'''
        CALL db.index.fulltext.queryNodes("{index_name}", "{search_query}") YIELD node, score
        WITH node, score AS search_score
        RETURN {return_statement}, search_score
        ORDER BY search_score DESC LIMIT {top_k}
        '''
        res = self.db_client.execute_query(
            query_=query,
            routing_=RoutingControl.READ,
            result_transformer_ = lambda r: r.data()
        )
        return json.dumps(res, indent=4)

    def node_embedding_search(self,
                              search_query:str,
                              node_label: str,
                              search_prop:str,
                              top_k=10) -> str:

        index_name = self.schema.schema.get_node_search_field(node_label, search_prop, "TEXT_EMBEDDING").indexName
        return_props = self.schema.schema.get_node_properties(node_label)
        return_statement = ', '.join([f'node.`{p}` AS `{p}`' for p in return_props])
        query_vector = self.data.embedding_model.embed_query(search_query)
        query = f'''
        CALL db.index.vector.queryNodes('{index_name}', {top_k}, $queryVector) YIELD node, score
        WITH node, score AS search_score
        RETURN {return_statement}, search_score
        ORDER BY search_score DESC LIMIT {top_k}
        '''
        res = self.db_client.execute_query(
            query_=query,
            routing_=RoutingControl.READ,
            result_transformer_ = lambda r: r.data(),
            queryVector = query_vector
        )
        return json.dumps(res, indent=4)

    def node_search(self, search_config:Dict[str,str], search_query:str, top_k=10):
        """
        Performs a search operation on nodes using different modes such as full-text
        or semantic searches. The method executes the search by delegating
        operations to the respective helper functions based on the specified search
        type.

        Parameters:
            search_config (Dict[str, str]): A dictionary specifying the configuration
                for the search operation. It must contain the following keys:
                    - "search_type": Determines the type of search to perform, either
                      "FULLTEXT" for full-text search or "SEMANTIC" for embedding-based
                      search.
                    - "node_label": The label of the node to search within the graph.
                    - "search_prop": The property of the node to search against.

            search_query (str): The query string used to perform the search

            top_k (int): The maximum number of results to return. Defaults to 10.

        Returns:
            The results of the performed search operation, as provided by the
            corresponding helper function.

        """

        if search_config['search_type'] == "FULLTEXT":
            return self.node_fulltext_search(search_query, search_config['node_label'], search_config['search_prop'], top_k)
        elif search_config['search_type'] == "SEMANTIC":
            return self.node_embedding_search(search_query, search_config['node_label'], search_config['search_prop'], top_k)
        else:
            raise ValueError(f"Invalid search type: {search_config['search_type']}")

    def query(self, query_instructions:str) -> str:
        """
        Traverses a graph database based on specific instructions. The method formulates a traversal query
        based on the given query instructions and executes it using a database client.
        The results of the execution are transformed into JSON format and returned as a string.

        Arguments:
            query_instructions: A string containing detailed instructions for the query.

        Returns:
            str: The JSON formatted result of the executed query.

        Raises:
            KeyError: If a required key is missing during template invocation.
            DatabaseExecutionError: If the query execution fails in the database.
        """
        prompt  = QUERY_TEMPLATE.invoke({'queryInstructions': query_instructions,
                                         'graphSchema':self.schema.schema.prompt_str()})
        query = self.llm.invoke(prompt).content
        print(f'Running Query:\n{query}')
        res = self.db_client.execute_query(
            query_=query,
            routing_=RoutingControl.READ,
            result_transformer_ = lambda r: r.data(),
        )

        #remove embeddings as this can blow context window and makes thing hard to read
        for embedding in self.schema.schema.get_all_text_embedding_names():
            remove_key_recursive(res, embedding)
        #turn into formated string and return
        return json.dumps(res, indent=4)

    def aggregate(self, agg_instructions:str):
        """
        Aggregates data from a database based on specific instructions. The method formulates a query
        based on the given aggregation instructions and executes it using a database client.
        The results of the execution are transformed into JSON format and returned as a string.

        Parameters:
            agg_instructions (str): Instructions that detail how the data should be aggregated.
                These instructions will be used to generate the query.

        Returns:
            str: A JSON-formatted string representation of the aggregated data based on the executed query.

        Raises:
            Any exceptions that may occur during query formulation or execution will propagate and are
            not directly handled within this method.
        """
        prompt  = AGG_QUERY_TEMPLATE.invoke({'queryInstructions': agg_instructions,
                                             'graphSchema':self.schema.schema.prompt_str()})
        query = self.llm.invoke(prompt).content
        print(f'Running Query:\n{query}')
        res = self.db_client.execute_query(
            query_=query,
            routing_=RoutingControl.READ,
            result_transformer_ = lambda r: r.data(),
        )
        #remove embeddings as this can blow context window and makes thing hard to read
        for embedding in self.schema.schema.get_all_text_embedding_names():
            remove_key_recursive(res, embedding)
        #turn into formated string and return
        return json.dumps(res, indent=4)

    def _create_or_replace_internal_agent(self):
        tools = [self.node_search, self.query, self.aggregate]
        self.agent_executor = create_react_agent(self.llm, tools)

    def agent(self, question: str):
        """
        Answers a question using GraphRAG.

        This method relies on an internal prebuilt LangGraph ReAct agent which it creates if it doesn't already exist.
        This agent leverages the graph schema to customize tools prompts.

        Args:
            question (str): The question to be executed.

        """
        if not self.agent_executor:
            self._create_or_replace_internal_agent()
        system_instruction = INTERNAL_AGENT_SYSTEM_TEMPLATE.invoke({'searchConfigs':self.get_search_configs_prompt(),
                                                        'graphSchema':self.schema.schema.prompt_str()}).to_string()
        #print(system_instruction)
        for step in self.agent_executor.stream(
                {"messages": [SystemMessage(content=system_instruction), HumanMessage(content=question)]},
                stream_mode="values",
        ):
            step["messages"][-1].pretty_print()

    def create_react_agent(self, **kwargs):
        """
        A factory for creating Langgraph Agents backed with GraphRAG and Knowledge Graph

        Arguments:
            **kwargs: Keyword arguments passed to the original `create_react_agent`.

        Returns:
            The result of invoking `create_react_agent`.
        """
        if "tools" in kwargs:
            other_tools = kwargs["tools"]
            if not isinstance(other_tools, Sequence):
                raise ValueError(f"'tools' must be a Sequence, but got {type(other_tools).__name__} instead.")
            for index, item in enumerate(other_tools):
                if not isinstance(item, (BaseTool, Callable)):
                    raise ValueError(
                        f"Invalid item at index {index}: {item} (type: {type(item).__name__}). "
                        f"'tools' must only contain instances of BaseTool or Callable."
                    )
        else:
            other_tools = []

        # Combine tools
        all_tools = [self.node_search, self.query, self.aggregate] + other_tools

        # Inject the combined tools into `kwargs`
        kwargs["tools"] = all_tools

        # Inject prompt into kwargs
        if "prompt" in kwargs:
            if kwargs["prompt"] is None or not isinstance(kwargs["prompt"], str):
                raise ValueError("`prompt` must be a non-null string.")
            additional_instruction = "## Additional Instructions\n" + kwargs["prompt"]
        else:
            additional_instruction = ""  # Default to an empty string if `prompt` key is not in kwargs

        # Get model if not included
        if "model" not in kwargs:
            kwargs["model"] = self.llm

        kwargs["prompt"] = AGENT_SYSTEM_TEMPLATE.invoke({'searchConfigs':self.get_search_configs_prompt(),
                                                           'graphSchema':self.schema.schema.prompt_str(),
                                                           'additionalInstructions': additional_instruction}).to_string()

        return create_react_agent(**kwargs)