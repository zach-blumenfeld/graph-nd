from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from graph_nd.graphrag.graph_schema import GraphSchema

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

class PropertyMapping(BaseModel):
    columnName: str = Field(description="the column name from the table that is to be mapped map to a property")
    propertyName: str = Field(description="the property name to be mapped to")

class NodeMap(BaseModel):
    """
    The mapping of table columns to a node id and properties.
    """
    nodeLabel: str = Field(description="the node label")
    nodeId: PropertyMapping  = Field(description="the node id mapping")
    properties: Optional[List[PropertyMapping]] = Field(None, description="A mapping of other table column names to node property names.")

    def convert_to_node_record(self, table_record: Dict) -> Dict:
        node_record = {self.nodeId.propertyName: table_record[self.nodeId.columnName]}
        if self.properties:
            for prop in self.properties:
                node_record[prop.propertyName] = table_record[prop.columnName]
        return {'label':self.nodeLabel, 'record': node_record}

class RelationshipMap(BaseModel):
    """
    The mapping of table columns to a relationship and its start and end nodes.
    """
    relationshipType: str = Field(description="the relationship type")
    relationshipId: PropertyMapping = Field(None, description="the relationship id mapping, if applicable")
    properties: Optional[List[PropertyMapping]] = Field(None, description="A mapping of other table column names to relationship property names.")
    startNodeMap: NodeMap = Field(description="the node map for the start node")
    endNodeMap: NodeMap = Field(description="the node map for the end node")
    def convert_to_rel_record(self, table_record: Dict) -> Dict:
        rel_record = {
            'start_node_id':table_record[self.startNodeMap.nodeId.columnName],
            'end_node_id': table_record[self.endNodeMap.nodeId.columnName],
        }
        if self.relationshipId:
            rel_record[self.relationshipId.propertyName] = table_record[self.relationshipId.columnName]
        if self.properties:
            for prop in self.properties:
                rel_record[prop.propertyName] = table_record[prop.columnName]
        return {'rel_type':self.relationshipType,
                'start_node_label':self.startNodeMap.nodeLabel,
                'end_node_label':self.endNodeMap.nodeLabel,
                'record': rel_record}

    def convert_to_triple_record(self, table_record: Dict) -> Tuple[Dict, Dict, Dict]:
        return (self.startNodeMap.convert_to_node_record(table_record),
                self.convert_to_rel_record(table_record),
                self.startNodeMap.convert_to_node_record(table_record))


class RelTableMapping(BaseModel):
    """
    The mapping of table columns to graph relationships.
    """
    tableName: str = Field(..., description="the name of the table")
    tableDescription: str = Field(default='', description="description of the table")
    relationshipMaps: List[RelationshipMap] = Field(description="the relationships and their nodes that this table maps to")

    def convert_to_triple_records(self, table_record: Dict) -> Dict[str, Tuple[Dict, Dict, Dict]]:
       triple_records = dict()
       for re_map in self.relationshipMaps:
           triple = re_map.convert_to_triple_record(table_record)
           triple_records[f"({triple[0]['label']})-[{triple[1]['rel_type']}]->({triple[2]['label']})"] = triple
       return triple_records



class NodeTableMapping(NodeMap):
    """
    The mapping of table columns to a graph node.
    """
    tableName: str = Field(..., description="the name of the table")
    tableDescription: str = Field(default='', description="description of the table")