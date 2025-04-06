import asyncio
import json
from typing import Optional, List

from neo4j import Driver
from pydantic import Field, BaseModel
from tqdm.asyncio import tqdm as tqdm_async

from graph_nd.ai_tier.data_source import DataSource, SourceMappings, SourceMappingDirectives, SourceMapping, \
    SourceMappingDirective, LLMTransformType, TextMapping
from graph_nd import GraphRAG
from graph_nd.ai_tier.prompt_templates import SCHEMA_MAPPING_DIRECTIVES_TEMPLATE, \
    RELATIONSHIPS_MAPPING_FROM_DIR_TEMPLATE, NODE_MAPPING_FROM_DIR_TEMPLATE
from graph_nd.graphrag.source_metadata import TransformType
from graph_nd.graphrag.table_mapping import NodeTableMapping, RelTableMapping
from graph_nd.graphrag.utils import run_async_function


class TrueOrFalse(BaseModel):
    """
    Binary true or false response
    """
    answer: bool = Field(..., description="True or False")

# Models class to package LLMs and Embedders. This will be made more sophisticated later
class Models:
    def __init__(self, llm, embedder, llm_knowledge_mapping=None):
        self.llm = llm
        self.embedder = embedder
        self.llm_knowledge_mapping = (
            llm_knowledge_mapping
            if llm_knowledge_mapping
            else llm
        )


class AITier:
    def __init__(self, models: Models, knowledge_graph: Driver, data_sources: Optional[List[DataSource]] = None):

        # Set data sources
        self.data_sources = {}
        if data_sources:
            self.add_data_sources(data_sources)

        # Initialize core components
        self.agent = self.Agent(self, models=models)
        self.knowledge = self.Knowledge(self, graph_client=knowledge_graph, models=models)


    def add_data_source(self, data_source: DataSource):
        """Add a data source to the map."""
        if not isinstance(data_source, DataSource):
            raise TypeError("All data sources must be instances of DataSource.")
        unique_name = data_source.unique_name()
        if unique_name in self.data_sources:
            raise ValueError(f"DataSource with unique name '{unique_name}' already exists.")
        self.data_sources[unique_name] = data_source

    def get_data_source(self, unique_name: str) -> DataSource:
        """Retrieve a data source by its unique name."""
        return self.data_sources.get(unique_name)

    def list_data_sources(self) -> List[str]:
        """List all unique names of registered data sources."""
        return list(self.data_sources.keys())

    def add_data_sources(self, sources: List[DataSource]):
        """
        Adds additional data sources to the tier.
        """
        for source in sources:
            self.add_data_source(source)


    # Nested DataSources Class

    # Nested Knowledge Class
    class Knowledge:
        def __init__(self, tier: "AITier", graph_client: Driver, models: Models):
            """
            Initializes the Knowledge interface, including all graph-related functionality.
            """
            self.graphrag = GraphRAG(
                db_client=graph_client,
                llm=models.llm,
                embedding_model=models.embedder
            )
            self.llm_mapping_directives = models.llm_knowledge_mapping.with_structured_output(SourceMappingDirectives, method="function_calling")
            self.llm_node_table_mapping = models.llm_knowledge_mapping.with_structured_output(NodeTableMapping, method="function_calling")
            self.llm_rel_table_mapping = models.llm_knowledge_mapping.with_structured_output(RelTableMapping,
                                                                                              method="function_calling")
            self.data_sources: dict[str, DataSource] = tier.data_sources
            self.mappings: SourceMappings = SourceMappings(mappings=[])

        def sync(self):
            """
            Syncs all data sources into the knowledge graph.
            """
            # for source in self.data_sources:
            #     if source.type() == DataSourceType.RDBMS:
            #         schema = source.schema()
            #         for table_name in schema.get("tables", []):
            #             data = source.get(table_name)
            #             self.graphrag.data.merge_node_table(data)
            #     elif source.type() == DataSourceType.TEXT_DOCUMENTS:
            #         # Handle text document merging
            #         for doc_name in source.storage_client.list_files():
            #             doc_text = source.get(doc_name)
            #             self.graphrag.data.merge_texts([doc_text])

        async def create_node_table_mapping_from_directive(self, source_mapping_directive: SourceMappingDirective) -> NodeTableMapping:
            name = source_mapping_directive.entity_name
            table_schema = self.data_sources[source_mapping_directive.data_source_name].get_table_schema(name).model_dump_json(indent=4)
            prompt = NODE_MAPPING_FROM_DIR_TEMPLATE.invoke({'tableName': name,
                                                   'directions': source_mapping_directive.mapping_directions,
                                                   'tableSchema': table_schema,
                                                   'graphSchema': self.graphrag.schema.schema.prompt_str()})
            mapping: NodeTableMapping = await self.llm_node_table_mapping.ainvoke(prompt)
            return mapping

        async def create_node_rel_table_mapping_from_directive(self, source_mapping_directive: SourceMappingDirective) -> RelTableMapping:
            name = source_mapping_directive.entity_name
            table_schema = self.data_sources[source_mapping_directive.data_source_name].get_table_schema(name).model_dump_json(indent=4)
            prompt = RELATIONSHIPS_MAPPING_FROM_DIR_TEMPLATE.invoke({'tableName': name,
                                                   'directions': source_mapping_directive.mapping_directions,
                                                   'tableSchema': table_schema,
                                                   'graphSchema': self.graphrag.schema.schema.prompt_str()})
            mapping: RelTableMapping = await self.llm_rel_table_mapping.ainvoke(prompt)
            return mapping

        async def create_mapping_from_directive(self, source_mapping_directive: SourceMappingDirective, semaphore) -> SourceMapping:
            async with semaphore:
                match source_mapping_directive.mapping_type:
                    case LLMTransformType.LLM_TABLE_MAPPING_TO_NODE:
                        mapping = await self.create_node_table_mapping_from_directive(source_mapping_directive)
                        mapping_type = TransformType.LLM_TABLE_MAPPING_TO_NODE
                    case LLMTransformType.LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS:
                        mapping = await self.create_node_rel_table_mapping_from_directive(source_mapping_directive)
                        mapping_type = TransformType.LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS
                    case LLMTransformType.LLM_TEXT_EXTRACTION_TO_NODES:
                        #TODO: We have no way to filter nodes in entity extraction...likely important -> extend in GraphRAG
                        mapping = TextMapping(nodesOnly=True)
                        mapping_type = TransformType.LLM_TEXT_EXTRACTION_TO_NODES
                    case LLMTransformType.LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS:
                        mapping = TextMapping(nodesOnly=False)
                        mapping_type = TransformType.LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS
                return SourceMapping(
                    data_source_name=source_mapping_directive.data_source_name,
                    entity_name=source_mapping_directive.entity_name,
                    mapping_type=mapping_type,
                    mapping=mapping
                )

        async def create_mappings_from_directives(self, source_mapping_directives: SourceMappingDirectives, max_workers=10) -> SourceMappings:
            # Create a semaphore with the desired number of workers
            semaphore = asyncio.Semaphore(max_workers)
            # Create tasks with the semaphore
            tasks = [self.create_mapping_from_directive(smd, semaphore) for smd in source_mapping_directives.source_mapping_directives]
            # Explicitly update progress using `tqdm` as tasks complete
            results = []
            with tqdm_async(total=len(tasks), desc="Creating Source Mappings From Directives") as pbar:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    results.append(result)
                    pbar.update(1)  # Increment progress bar for each completed task

            return SourceMappings(mappings=results)

        def infer_graph_schema(self, use_cases:str):
            data_source_models = [{"dataSourceName":name, "dataSourceSchema":ds.schema().model_dump()} for name, ds in self.data_sources.items()]
            data_source_str = json.dumps(data_source_models, indent=4)
            self.graphrag.schema.infer_from_use_case(use_cases, data_source_str)

        def infer_mapping_directives(self, use_cases:str) -> SourceMappingDirectives:
            data_source_models = [{"dataSourceName":name, "dataSourceSchema":ds.schema().model_dump()} for name, ds in self.data_sources.items()]
            prompt = SCHEMA_MAPPING_DIRECTIVES_TEMPLATE.invoke({'useCase':use_cases,
                                                                'sourceDataModels':data_source_models,
                                                                'graphSchema': self.graphrag.schema.prompt_str()})
            mappings_directives = self.llm_mapping_directives.invoke(prompt)
            return mappings_directives

        def infer_mapping(self, use_cases:str, max_workers=10):

            # infer graph schema
            self.infer_graph_schema(use_cases)

            # get mapping directives
            mappings_directives = self.infer_mapping_directives(use_cases)

            #create mappings
            self.mappings = run_async_function(self.create_mappings_from_directives, mappings_directives, max_workers)

    # Nested Agent Class
    class Agent:
        def __init__(self, ai_tier: "AITier", models: Models):
            """
            Initializes the Agent interface for interacting with the AI Tier's knowledge graph and external tools.
            """
            self.parent = ai_tier  # Reference to AiTier for shared context
            self.models = models
            self.data_source_map: dict[str, DataSource] = ai_tier.data_sources


        def invoke(self, question: str) -> str:
            """
            Invokes the agent to handle a question using the knowledge graph.
            """
            return self.parent.knowledge.graphrag.agent(question)

        def add_tool(self, tool_config: dict):
            """
            Adds a tool configuration to the agent.
            """
            pass

