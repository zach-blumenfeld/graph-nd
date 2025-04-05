from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Any
from enum import Enum

import pandas as pd
import snowflake
from pydantic import BaseModel
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkClientException

from graphrag.source_metadata import SourceType, TransformType
from graphrag.table_mapping import NodeTableMapping, RelTableMapping


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

class SourceMapping(BaseModel):
    source_name: str
    mapping_type: TransformType
    mapping: Union[TextMapping, NodeTableMapping, RelTableMapping]

# Abstract Base Class for DataSource
class DataSource(ABC):

    @abstractmethod
    def schema(self) -> SourceSchema:
        """
        Retrieves the schema of the data source
        """
        pass

    @abstractmethod
    def get_table(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves a table from the data source.
        """
        pass

    @abstractmethod
    def get_text_doc(self, name: str) -> str:
        """
        Retrieves a text document from the data source.
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
            table_name = table["name"]
            table_comment = table["comment"] or ""  # Use comment if available or default to empty

            # Describe the table structure using Pandas
            fields = self.session.sql(f"DESCRIBE TABLE {table_name}").collect()
            fields_df = pd.DataFrame([row.as_dict() for row in fields])

            # Only keep relevant columns and convert to dict
            table_schema = fields_df[["name", "type", "primary key", "unique key", "comment"]].to_dict(orient="records")

            # Add SourceEntitySchema for the table
            entities.append(SourceEntitySchema(
                name=table_name,
                description=table_comment,
                type=SourceType.STRUCTURED_TABLE_RDBMS,
                entity_schema=table_schema
            ))

        # Step 3: Return a SourceSchema including database details
        return SourceSchema(
            name=f"{self.config['database'].upper()}.{self.config['schema'].upper()}",
            description=schema_comment,
            entities=entities
        )

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

    def get_text_doc(self, name: str) -> str:
        """
        SnowflakeDB is not designed for text documents. Raise an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support get_text_doc().")

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

    def close(self):
        """
        Closes the Snowflake session.
        """
        self.session.close()




# class S3TextDocs(DataSource):
#     """
#     Data Source for an S3 bucket containing text (.txt or .pdf) documents.
#     """