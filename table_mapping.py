from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from GraphSchema import GraphSchema

class TableTypeEnum(str, Enum):
    SINGLE_NODE = 'SINGLE_NODE'
    RELATIONSHIPS = 'RELATIONSHIPS'

class TableType(BaseModel):
    """
    The type of table. Either
        - SINGLE_NODE: a table that maps to a single entity i.e. node
        - RELATIONSHIPS: a table that maps to 1 or more relationships between entities
    """
    type: TableTypeEnum = Field(description="the type of the table")

class NodeMap(BaseModel):
    """
    The mapping of table columns to a node id and properties.
    """
    nodeLabel: str = Field(description="the node label")
    nodeIdColumn: str  = Field(description="the column name that maps to the node id property")
    properties: Optional[Dict[str, str]] = Field(None, description="A map of other table column names (keys) to node property names (values).")

class RelationshipMap(BaseModel):
    """
    The mapping of table columns to a relationship and its start and end nodes.
    """
    relationshipType: str = Field(description="the relationship type")
    relationshipIdColumn: Optional[str]  = Field(None, description="the column name that maps to the relationship id property if applicable")
    properties: Optional[Dict[str, str]] = Field(description="A map of other table column names (keys) to relationship property names (values).")
    startNodeMap: NodeMap = Field(description="the node map for the start node")
    endNodeMap: NodeMap = Field(description="the node map for the end node")

class RelTableMapping(BaseModel):
    """
    The mapping of table columns to graph relationships.
    """
    tableName: str = Field(description="the name of the table")
    tableDescription: str = Field(description="description of the table")
    relationshipMaps: List[RelationshipMap] = Field(description="the relationships and their nodes that this table maps to")

class NodeTableMapping(NodeMap):
    """
    The mapping of table columns to a graph node.
    """
    tableName: str = Field(description="the name of the table")
    tableDescription: str = Field(description="description of the table")


class RelTableConverter:
    def __init__(self, rel_table_mappings: List[RelTableMapping], graph_schema: GraphSchema):
        self.rel_table_mappings = rel_table_mappings
        self.graph_schema = graph_schema

    def convert_to_node_records(self, table_records:List[Dict]) -> List[Dict]:
        return [dict(), dict()]
    def convert_to_relationship_records(self, table_records:List[Dict]) -> List[Dict]:
        return [dict(), dict()]

class NodeTableConverter:
    def __init__(self, node_table_mapping: NodeMap, graph_schema: GraphSchema):
        self.node_table_mapping = node_table_mapping
        self.graph_schema = graph_schema

    def convert_to_node_records(self, table_records:List[Dict]) -> List[Dict]:
        return [dict(), dict()]
    def convert_to_relationship_records(self, table_records:List[Dict]) -> List[Dict]:
        return [dict(), dict()]