from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Any
from enum import Enum

import pandas as pd
import snowflake
from pydantic import BaseModel, Field
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkClientException

from graph_nd.graphrag.source_metadata import SourceType, TransformType
from graph_nd.graphrag.table_mapping import NodeTableMapping, RelTableMapping


class SourceEntitySchema(BaseModel):
    name: str
    description: str
    type: SourceType
    entity_schema: Any

class SourceSchema(BaseModel):
    name: str
    description: str
    entities: List[SourceEntitySchema]

#TODO: We need a way to subset the graph schema
class TextMapping(BaseModel):
    nodesOnly: bool
    #subsets

class LLMTransformType(Enum):
    """
    Enumeration that represents different types of transformations or mappings
    that can be performed on source data by LLMs.
    Provide a standardized way to describe and categorize these types of transformations.
    Attributes:
        LLM_TABLE_MAPPING_TO_NODE (str): A table that maps to a single entity (i.e. node) from the graphSchema
        LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS (str): A table that maps to one or more relationships between nodes in the graphSchema
        LLM_TEXT_EXTRACTION_TO_NODES (str): Text content that maps to one or more nodes in the graphSchema
        LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS (str): Text that maps to nodes and one or more relationships between nodes in the graphSchema
    """
    LLM_TABLE_MAPPING_TO_NODE = "TABLE_MAPPING_TO_NODE"
    LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS = "TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS"
    LLM_TEXT_EXTRACTION_TO_NODES = "TEXT_EXTRACTION_TO_NODES"
    LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS = "TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS"

class SourceMappingDirective(BaseModel):
    """
    Specifies a detailed directive for mapping data from a data source entity to the target graphSchema.
    """
    data_source_name: str = Field(..., description="The unique name of the data source from where the entity originates. Used to retrieve from the data source index later.")
    entity_name: str = Field(..., description="The specific name of the entity such as the table name or name of text body document. Used to retrieve the entity index later.")
    mapping_type: LLMTransformType = Field(..., description="Type of transformation/mapping to apply")
    mapping_directions: str = Field(..., description="detailed description of how the entity should be "
                                                     "transformed or mapped to the target. "
                                                     "Another LLM will conduct the mapping by only reading the "
                                                     "GraphSchema and this SourceMappingDirective in isolation."
                                                     "The description must be detailed enough for the LLM to conduct the mapping correctly "
                                                     "including target node labels, relationship types, and properties.")

class SourceMappingDirectives(BaseModel):
    """
    Collection of source mapping directives.
    """
    source_mapping_directives: List[SourceMappingDirective] = Field(..., description="List of SourceMappingDirectives")

#TODO: We need a way to subset the graph schema for text mapping -> extend graphrag for this
class SourceTextMapping(BaseModel):
    data_source_name: str
    entity_name: str
    nodes_only: bool
    # subset

class SourceMapping(BaseModel):
    data_source_name: str
    entity_name: str
    mapping_type: TransformType
    mapping: Union[TextMapping, NodeTableMapping, RelTableMapping]

class SourceMappings(BaseModel):
    mappings: List[SourceMapping]


# Abstract Base Class for DataSource
class DataSource(ABC):

    @abstractmethod
    def schema(self) -> SourceSchema:
        """
        Retrieves the schema of the data source
        """
        pass

    @abstractmethod
    def get_table_schema(self, name: str) -> SourceEntitySchema:
        """
        Retrieves a table schema from the data source.
        """
        pass

    @abstractmethod
    def get_table(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves a table from the data source.
        """
        pass

    @abstractmethod
    def get_text_doc(self, name: str) -> List[str]:
        """
        Retrieves a text document from the data source.
        Currently, DataSource Implementation is responsible for chunking strategy
        """
        pass

    @abstractmethod
    def get_mapping(self, name: str) -> SourceMapping:
        """
        Retrieves a mapping given the source name
        """
        pass

    @abstractmethod
    def get_mappings(self) -> List[SourceMapping]:
        """
        Retrieves all the source mappings
        """
        pass

    @abstractmethod
    def set_mapping(self, mapping: SourceMapping):
        """
        Sets the source mapping
        """
        pass

    @abstractmethod
    def unique_name(self) -> str:
        """
        Generate a unique name for the data source.
        Subclasses must implement this method to define what makes the data source unique.
        """
        pass


# noinspection SqlNoDataSourceInspection
class SnowflakeDB(DataSource):
    """
    Data Source for a Snowflake Database using Snowpark.
    """

    def __init__(self, account: str, user: str, password: str, database: str, schema: str, warehouse: str, role: str):
        """
        Initialize the SnowflakeDB data source with Snowpark Session parameters.
        """
        self.config = {
            "account": account,
            "user": user,
            "password": password,
            "database": database,
            "schema": schema,
            "warehouse": warehouse,
            "role": role
        }
        self.session = Session.builder.configs(self.config).create()
        self.mappings = {}  # To store source mappings

    def _get_table_schema(self, name: str) -> SourceEntitySchema:
        # Describe the table structure using Pandas
        fields = self.session.sql(f"DESCRIBE TABLE {name}").collect()
        fields_df = pd.DataFrame([row.as_dict() for row in fields])

        # Only keep relevant columns and convert to dict
        table_schema = fields_df[["name", "type", "primary key", "unique key", "comment"]].to_dict(orient="records")

        # Add SourceEntitySchema for the table
        return SourceEntitySchema(
            name=name,
            description='',
            type=SourceType.STRUCTURED_TABLE_RDBMS,
            entity_schema=table_schema
        )

    def schema(self) -> SourceSchema:
        """
        Retrieve and describe the schema of the current database.
        :return: Detailed SourceSchema with table and field-level metadata
        """
        # Step 1: Retrieve current snow schema details
        snow_schemas = self.session.sql(f"SHOW SCHEMAS IN {self.config['database']}").collect()

        # Identify the current database schema and its comment
        snow_schema_info = next(
            (sch for sch in snow_schemas if sch["name"].upper() == self.config['schema'].upper()),
            None
        )

        if not snow_schema_info:
            raise ValueError("Schema not found in the list of database schemas.")
        schema_comment = snow_schema_info.comment or "No description available."

        # Step 2: Retrieve all tables in the current schema
        tables_rows = self.session.sql("SHOW TABLES").collect()
        tables_df = pd.DataFrame([row.as_dict() for row in tables_rows])

        # Filter to the relevant columns 'name' and 'comment'
        tables_df = tables_df[["name", "comment"]]


        entities = []

        for _, table in tables_df.iterrows():
            entity_schema = self._get_table_schema(table["name"])

            entity_schema.description = table["comment"] or ""  # Use comment if available or default to empty

            # Add SourceEntitySchema for the table
            entities.append(entity_schema)

        # Step 3: Return a SourceSchema including database details
        return SourceSchema(
            name=f"{self.config['database'].upper()}.{self.config['schema'].upper()}",
            description=schema_comment,
            entities=entities
        )

    def get_table_schema(self, name: str) -> SourceEntitySchema:
        entity_schema = self._get_table_schema(name)
        table_comment_query = f"""
            SELECT COMMENT FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME='{name.upper()}' AND  TABLE_SCHEMA='{self.config['schema'].upper()}'
        """
        comment_result = self.session.sql(table_comment_query).collect()
        entity_schema.description = comment_result[0]["COMMENT"] if comment_result else ""
        return entity_schema

    def get_table(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves a table from Snowflake.
        """
        try:
            # Fetch the entire table as a pandas DataFrame
            df = self.session.table(f"{self.config['schema']}.{name}").to_pandas()
            return df.to_dict(orient="records")
        except SnowparkClientException as e:
            raise RuntimeError(f"Failed to fetch table '{name}': {str(e)}")

    def get_text_doc(self, name: str) -> List[str]:
        """
        Not Yet Supported
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not yet support get_text_doc().")

    def get_mapping(self, name: str) -> SourceMapping:
        """
        Retrieves a mapping for a given table name.
        """
        if name not in self.mappings:
            raise ValueError(f"No mapping exists for table: {name}")
        return self.mappings[name]

    def get_mappings(self) -> List[SourceMapping]:
        """
        Retrieves all source mappings.
        """
        return list(self.mappings.values())

    def set_mapping(self, mapping: SourceMapping):
        """
        Sets a mapping for a given table.
        """
        table_name = mapping.table_name  # Assume mapping has an attribute 'table_name'
        self.mappings[table_name] = mapping

    def unique_name(self) -> str:
        # Compose a unique name using relevant attributes
        return f"Snowflake::{self.config['account']}.{self.config['database']}.{self.config['schema']}"

    def close(self):
        """
        Closes the Snowflake session.
        """
        self.session.close()




# class S3TextDocs(DataSource):
#     """
#     Data Source for an S3 bucket containing text (.txt or .pdf) documents.
#     """