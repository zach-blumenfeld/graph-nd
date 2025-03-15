from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import yaml  # Import PyYAML for YAML serialization



class Element(BaseModel):
    """
    Base class for graph elements
    """
    description: str = Field(description="description of this element including instructions for use in loading, query, and search")


class PropertySchema(Element):
    """
    A property of either a node or relationship
    """
    name: str= Field(description="name of the property")
    type: str = Field(description="data type of the property, STRING, INTEGER, etc.")


class NodeSchema(Element):
    """
    A graph node. Represents an entity (Person, Place, Thing... etc.)
    """
    id: PropertySchema = Field(description="the property to use as the unique non-null identifier for the node")
    label: str = Field(description="Type of the node (e.g., Person, Location, etc.). Should be Title CamelCase to conform to style standards.")
    properties: List[PropertySchema] = Field(
        default_factory=list, description="Other properties for the node. must include at least the key property"
    )

class QueryPattern(Element):
    """
   A query pattern for a relationship describing a start and end node i.e (startNode)-[r]->(endNode)
    """
    startNode:str = Field(description="Starting node label of the relationship query pattern")
    endNode: str = Field(description="Ending node label of the relationship query pattern")


class RelationshipSchema(Element):
    """
   A graph relationship. Represents actions or associations between entities
    """
    id: Optional[PropertySchema] = Field(None, description="optional property to use as the unique non-null identifier for the relationship.  "
                                      "only necessary for parallel relationship (more than one instance of a "
                                      "relationships of the same type between the same start and end nodes.  ")
    type: str = Field(description="The relationship type.  Should be in all caps to conform to style standards.")
    queryPatterns: List[QueryPattern] = Field(default_factory=list, description="Query patterns for the relationship")
    properties: List[PropertySchema] = Field(
        default_factory=list, description="Properties for the relationship. must include at least the key property"
    )

    def query_model_dump(self, **kwargs) -> dict:
        """
        Custom dict method to serialize query patterns in the format:
        (:startNodeLabel)-[:TYPE]->(:endNodeLabel)
        """
        # Generate the base dictionary from the parent method
        base_dict = super().model_dump(**kwargs)

        # Customize queryPatterns serialization
        if self.queryPatterns:
            base_dict["queryPatterns"] = [
                f"(:{pattern.startNode})-[:{self.type}]->(:{pattern.endNode})"
                for pattern in self.queryPatterns
            ]

        return base_dict


class GraphSchema(Element):
    nodes: List[NodeSchema] = Field(default_factory=list, description="List of nodes in the graph")
    relationships: List[RelationshipSchema] = Field(default_factory=list, description="List of relationships in the graph")

    def query_model_dump(self, **kwargs) -> dict:
        """
        Custom model_dump for GraphSchema that ensures nested elements are
        serialized using their own query_model_dump logic.
        """
        base_dict = super().model_dump(**kwargs)

        # Serialize nodes and relationships explicitly to invoke custom logic
        base_dict["nodes"] = [node.model_dump(**kwargs) for node in self.nodes]
        base_dict["relationships"] = [relationship.query_model_dump(**kwargs) for relationship in self.relationships]

        return base_dict

    def query_model_to_yaml(self, **kwargs) -> str:
        """
        Serialize the GraphSchema into a YAML string representation.
        Leverages `model_dump` to generate the dictionary and converts it to YAML.
        """
        # Use PyYAML's dump() to convert the dictionary into YAML
        return yaml.dump(self.query_model_dump(**kwargs), sort_keys=False)  # sort_keys=False keeps the field order consistent

