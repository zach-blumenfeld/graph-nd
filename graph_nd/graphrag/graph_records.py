import warnings
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from graph_nd.graphrag.graph_data import NodeData, RelationshipData, GraphData
from graph_nd.graphrag.graph_schema import GraphSchema


class Node(BaseModel):
    """
    Represents a node record with a unique identifier, label, and properties.
    """
    node_id: Any  = Field(description="the property to use as the unique non-null identifier for the node")
    label: str = Field(description="Type of the node (e.g., Person, Location, etc.). Should be Title CamelCase to conform to style standards.")
    properties: Optional[Dict[str, Any]] = Field(None, description="Other properties for the node as applicable.")

    def to_flat_record(self):
        return {
            "__SCHEMA_NODE_ID__": self.node_id,
            **(self.properties or {})
        }

class Relationship(BaseModel):
    """
    Represents a relationship record with its properties and identifiers.
    """
    type: str = Field(description="The relationship type.  Should be in all caps to conform to style standards.")
    start_node_label: str = Field(description="Starting node label of the relationship query pattern")
    start_node_id: Any = Field(description="Starting node id of the relationship query pattern")
    end_node_label: str = Field(description="ending node label of the relationship query pattern")
    end_node_id: Any = Field(description="ending node id of the relationship query pattern")
    properties: Optional[Dict[str, Any]] = Field(None, description="Other properties for the relationship as applicable.")

    def to_flat_record(self):
        return {
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            **(self.properties or {})
        }
class SubGraphNodes(BaseModel):
    """
    Nodes to be merged into a knowledge graph.
    """
    nodes: List[Node] = Field(default_factory=list,
                                              description="Nodes in the subgraph.")
    def to_subgraph(self):
        return SubGraph(nodes=self.nodes, relationships=[])

class SubGraph(BaseModel):
    """
    A subgraph to be merged into a knowledge graph. including nodes and relationships.
    """
    nodes: List[Node] = Field(default_factory=list,
                                              description="Nodes in the subgraph.")
    relationships: Optional[List[Relationship]] = Field(None,
                                              description="Relationships in the subgraph. Every relationship must have start and node ids that exist in nodes.")

    #TODO: it would be better if GraphSchema owned validation and these record/subgraph classes were refactored
    # to be the same or parent classes of the GraphData ones. That would cut down on code make everything more consistent
    def convert_to_graph_data(self, graph_schema: GraphSchema) -> GraphData:
        """
        Converts SubGraph into valid GraphData using the provided GraphSchema for validation.

        Warnings will be raised if invalid nodes or relationships are encountered.

        Args:
            graph_schema (GraphSchema): The schema used for node and relationship validation.

        Returns:
            Tuple[List[NodeData], List[RelationshipData]]: Validated NodeData and RelationshipData objects.
        """

        # Validate nodes
        node_label_flat_records = dict()
        node_label_dfs = dict()

        # Create a map of node labels to DataFrame-like structures
        for node in self.nodes:
            node_label = node.label
            if node_label not in node_label_flat_records:
                node_label_flat_records[node_label] = []
            node_label_flat_records[node_label].append(node.to_flat_record())  # Assuming Node has a `dict()` method

        # Validate and convert nodes
        for label, records in node_label_flat_records.items():
            try:
                node_schema = graph_schema.get_node_schema_by_label(label)
            except KeyError:
                warnings.warn(
                    f"Node label '{label}' is not defined in the GraphSchema. Skipping nodes with this label.")
                continue  # Skip this node label

            df = pd.DataFrame(records)
            df = df.dropna(subset=["__SCHEMA_NODE_ID__"])
            df[node_schema.id.name] = df["__SCHEMA_NODE_ID__"]
            node_label_dfs[label] = df

        # Validate relationships
        rel_type_flat_records = dict()
        rel_triple_dfs = dict()

        # Create a map of relationship types (edges) to DataFrame-like structures
        if not self.relationships:
            self.relationships = []
        for relationship in self.relationships:
            rel_triple = (relationship.start_node_label, relationship.type, relationship.end_node_label)
            if rel_triple not in rel_type_flat_records:
                rel_type_flat_records[rel_triple] = []
            rel_type_flat_records[rel_triple].append(relationship.to_flat_record())

        # Validate and convert relationships
        for rel_triple, records in rel_type_flat_records.items():
            try:
                rel_schema = graph_schema.get_relationship_schema(rel_triple[1], rel_triple[0], rel_triple[2])
            except KeyError:
                warnings.warn(
                    f"Relationship with pattern ({rel_triple[0]})-[{rel_triple[1]}]->({rel_triple[2]}) "
                    f"is not defined in the GraphSchema. Skipping these relationships.")
                continue  # Skip this relationship type
            df = pd.DataFrame(records)
            # Filter out invalid relationships where start/end node IDs are missing
            df = df.dropna(subset=["start_node_id", "end_node_id"])

            #TODO: I Don't think we need this validation anymore after changing relationship merge
            # from MATCH-MATCH-MERGE to MERGE-MERGE-MERGE Pattern
            # Should research more to be sure
            ## Ensure that start/end node IDs exist in the validated NodeData
            #valid_start_node_ids = node_label_dfs[rel_triple[0]]["__SCHEMA_NODE_ID__"].to_list()
            #df = df[df['start_node_id'].isin(valid_start_node_ids)]
            #valid_end_node_ids = node_label_dfs[rel_triple[2]]["__SCHEMA_NODE_ID__"].to_list()
            #df = df[df['end_node_id'].isin(valid_end_node_ids)]
            #add validated relations
            rel_triple_dfs[rel_triple] = df

        # get node_datas and remove extra node id columns used for mapping
        node_data_list = []
        for label, df in node_label_dfs.items():
            df.drop(columns=["__SCHEMA_NODE_ID__"], inplace=True)
            node_data_list .append(NodeData(node_schema=graph_schema.get_node_schema_by_label(label),
                                            records=df.to_dict("records")))

        # get rel_datas and remove extra node id columns used for mapping
        relationship_data_list = []
        for rel_triple, df in rel_triple_dfs.items():
            rel_schema = graph_schema.get_relationship_schema(rel_triple[1], rel_triple[0], rel_triple[2])
            start_node_schema = graph_schema.get_node_schema_by_label(rel_triple[0])
            end_node_schema = graph_schema.get_node_schema_by_label(rel_triple[2])
            relationship_data_list.append(RelationshipData(rel_schema=rel_schema,
                                                          start_node_schema=start_node_schema,
                                                          end_node_schema=end_node_schema,
                                                          records=df.to_dict("records")))

        graph_data = GraphData(nodeDatas=node_data_list, relationshipDatas=relationship_data_list)
        return graph_data



