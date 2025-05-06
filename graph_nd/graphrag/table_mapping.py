from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

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


class NodeMapping(BaseModel):
    """
    The mapping of table columns to a node
    """
    name: str = Field('', description="the name of the table")
    label: str = Field(description="the node label")
    nodeId: Optional[Union[str, Tuple[str, str]]]  = Field(description="the node id mapping")
    properties: Optional[Dict[str, str]]= Field(None, description="A mapping of other table column names to node property names.")

    def to_table_mapping(self) -> NodeTableMapping:
        if self.properties:
            properties = []
            for k,v in self.properties.items():
                properties.append(PropertyMapping(columnName=k, propertyName=v))
        else:
            properties = None

        if isinstance(self.nodeId, str):
            node_id = PropertyMapping(columnName=self.nodeId, propertyName=self.nodeId)
        elif isinstance(self.nodeId, tuple):
            node_id = PropertyMapping(columnName=self.nodeId[0], propertyName=self.nodeId[1])
        else:
            raise ValueError(f"Invalid nodeId value: {self.nodeId} must be `str` or `Tuple[str, str]`")
        return NodeTableMapping(tableName=self.name,
                                tableDescription='',
                                nodeLabel=self.label,
                                nodeId=node_id,
                                properties=properties)


class RelMapping(BaseModel):
    """
    The mapping of table columns to a relationship
    """
    name: str = Field('', description="the name of the table")
    relType: str = Field(description="the relationship type")
    relId: Optional[Union[str, Tuple[str, str]]] = Field(None, description="the relationship id mapping, if applicable")
    properties: Optional[Dict[str, str]] = Field(default=None, description="A mapping of other table column names to relationship property names.")
    startNode: NodeMapping = Field(description="the node map for the start node")
    endNode: NodeMapping = Field(description="the node map for the end node")

    def to_table_mapping(self) -> RelTableMapping:
        if self.properties:
            properties = []
            for k, v in self.properties.items():
                properties.append(PropertyMapping(columnName=k, propertyName=v))
        else:
            properties = None

        if self.relId:
            if isinstance(self.relId, str):
                rel_id = PropertyMapping(columnName=self.relId, propertyName=self.relId)
            elif isinstance(self.relId, tuple):
                rel_id = PropertyMapping(columnName=self.relId[0], propertyName=self.relId[1])
            else:
                raise ValueError(f"Invalid relId value: {self.nodeId} must be `str` or `Tuple[str, str]`")
            rel_map = RelationshipMap(relationshipType=self.relType,
                                      relationshipId=rel_id,
                                      properties=properties,
                                      startNodeMap=self.startNode.to_table_mapping(),
                                      endNodeMap=self.endNode.to_table_mapping())
        else:
            rel_map = RelationshipMap(relationshipType=self.relType,
                                      properties=properties,
                                      startNodeMap=self.startNode.to_table_mapping(),
                                      endNodeMap=self.endNode.to_table_mapping())

        return RelTableMapping(tableName=self.name, relationshipMaps=[rel_map])