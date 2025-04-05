from typing import Optional, List

from neo4j import Driver

from ai_tier.data_source import DataSource, DataSourceType
from graphrag import GraphRAG

# Models class to package LLMs and Embedders. This will be made more sophisticated later
class Models:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder




class AITier:
    def __init__(self, models: Models, knowledge_graph: Driver, data_sources: Optional[List[DataSource]] = None):
        self.data_sources = data_sources if data_sources else []

        # Initialize core components
        self.agent = self.Agent(self, models=models, data_sources=self.data_sources)
        self.knowledge = self.Knowledge(graph_client=knowledge_graph, models=models, data_sources=self.data_sources)


    def add_data_sources(self, sources: List[DataSource]):
        """
        Adds additional data sources to the tier.
        """
        for source in sources:
            if not isinstance(source, DataSource):
                raise TypeError("All data sources must be instances of DataSource.")
        self.data_sources.extend(sources)

    # Nested Knowledge Class
    class Knowledge:
        def __init__(self, graph_client: Driver, models: Models, data_sources: List[DataSource]):
            """
            Initializes the Knowledge interface, including all graph-related functionality.
            """
            self.graphrag = GraphRAG(
                db_client=graph_client,
                llm=models.llm,
                embedding_model=models.embedder
            )
            self.data_sources = data_sources
            self.mapping = None

        def sync(self):
            """
            Syncs all data sources into the knowledge graph.
            """
            for source in self.data_sources:
                if source.type() == DataSourceType.RDBMS:
                    schema = source.schema()
                    for table_name in schema.get("tables", []):
                        data = source.get(table_name)
                        self.graphrag.data.merge_node_table(data)
                elif source.type() == DataSourceType.TEXT_DOCUMENTS:
                    # Handle text document merging
                    for doc_name in source.storage_client.list_files():
                        doc_text = source.get(doc_name)
                        self.graphrag.data.merge_texts([doc_text])


        def infer_mapping(self, use_cases:str):
            # create a graph schema using graph rag
            self.graphrag.schema.infer(use_cases)
        # Nested Mapping Class
        class Mapping:
            def __init__(self, graphrag: GraphRAG):
                """
                Initializes the Mapping interface for schema inference and table mapping.
                """
                self.graphrag = graphrag

            def infer(self, use_case_description: str):
                """
                Infers graph and table mappings based on the use case description.
                """
                self.graphrag.schema.infer(use_case_description)

    # Nested Agent Class
    class Agent:
        def __init__(self, parent: "AITier"):
            """
            Initializes the Agent interface for interacting with the AI Tier's knowledge graph and external tools.
            """
            self.parent = parent  # Reference to AiTier for shared context

        def invoke(self, question: str) -> str:
            """
            Invokes the agent to handle a question using the knowledge graph.
            """
            return self.parent.knowledge.graphrag.agent(question)

        def add_tool(self, tool_config: dict):
            """
            Adds a tool configuration to the agent.
            """
            self.parent.knowledge.graphrag.create_or_replace_agent(tool_config)

