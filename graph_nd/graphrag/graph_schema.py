import json

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, Union, List, Tuple, Self
import yaml  # Import PyYAML for YAML serialization

from graph_nd.graphrag.utils import validate_list_type

NEO4J_PROPERTY_TYPES = {"STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "DATETIME"}
class SubSchema:
    def __init__(self, nodes: Union[str, List[str]] = None,
                 patterns: Union[Tuple[str, str, str], List[Tuple[str, str, str]]] = None,
                 relationships: Union[str, List[str]] = None,
                 description: str = None):
        """
        Encapsulates the information required to subset a graph schema and ensures proper validation
        and conversion for the provided input data. `SubSchema` is used in methods like `GraphSchema.subset` to
        describe the graph schema filtering criteria.

        Parameters:
        ----------
        nodes: Union[str, List[str]], optional
            A node or list of node labels to include in the subset. If provided, the node schemas
            corresponding to these nodes will be retrieved.

        patterns: Union[Tuple[str, str, str], List[Tuple[str, str, str]]], optional
            A pattern or list of patterns defining relationships to filter by. Each pattern is a
            tuple containing:
            - Start node label (str)
            - Relationship type (str)
            - End node label (str)

            The relevant node schemas and relationship schemas will be included in the subset.

        relationships: Union[str, List[str]], optional
            A relationship type or list of relationship types to include in the subset.
            All query patterns for the relationship type (and their start and end nodes) will be included in the subset.

        description: str, optional
            A custom description for the subsetted graph schema.
            If not provided, a default description may be generated based on the existing schema and provided subset criteria.

        Raises:
        ------
        ValueError
            If none of `nodes`, `patterns`, or `relationships` are provided.
        TypeError
            If any of the inputs are not of the expected type.
        """

        self.nodes = validate_list_type(nodes, str, "nodes")
        self.patterns = validate_list_type(patterns, tuple, "patterns")
        self.relationships = validate_list_type(relationships, str, "relationships")
        self.description = description

        # Ensure at least one of the inputs is provided
        if not self.nodes and not self.patterns and not self.relationships:
            raise ValueError("At least one of nodes, patterns, or relationships must be specified.")

class Element(BaseModel):
    """
    Base class for graph elements
    """
    description: Optional[str] = Field('',description="description of this element including instructions for use in loading, query, and search")

class PropertySchema(Element):
    """
    A property of either a node or relationship
    """
    name: str = Field(description="name of the property")
    type: str = Field(description="data type of the property, STRING, INTEGER, etc.")

    @field_validator("type")
    def validate_type(cls, v):
        if v.upper() not in NEO4J_PROPERTY_TYPES:
            raise ValueError(f"Invalid property type. Must be one of: {NEO4J_PROPERTY_TYPES}")
        return v.upper()

class SearchFieldSchema(Element):
    """
    A field used for semantic search such as for vector similarity or fulltext search
    """
    name: str= Field(description="name of the field")
    type: str = Field(description="type of field: TEXT_EMBEDDING, FULLTEXT")
    calculatedFrom: str = Field(description="name of the source property for this field")


class NodeSchema(Element):
    """
    A graph node. Represents an entity (Person, Place, Thing... etc.)
    """
    id: PropertySchema = Field(description="the property to use as the unique non-null identifier for the node")
    label: str = Field(description="Type of the node (e.g., Person, Location, etc.). Should be Title CamelCase to conform to style standards.")
    properties: List[PropertySchema] = Field(
        default_factory=list, description="Other properties for the node. must include at least the key property"
    )
    searchFields: List[SearchFieldSchema] = Field(default_factory=list, description="fields used for semantic search, sourced from properties.")

    @field_validator("properties")
    def validate_properties(cls, v: list[PropertySchema]) -> list[PropertySchema]:
        if not v:
            raise ValueError("properties must contain at least one property.")
        return v
    
    @model_validator(mode="after")
    def validate_id_property_in_properties(self) -> Self:
        if not any(p.name == self.id.name for p in self.properties):
            raise ValueError(f"properties must contain the key property. id: {self.id.name}")
        return self
    


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

    #TODO: We probably need better data structures to index rather than scanning lists...but we shouldn't be doing this a lot at runtime so not prioritizing currently
    def get_query_pattern(self, start_node_label:str, end_node_label:str) -> QueryPattern:
        if self.queryPatterns:
            for pattern in self.queryPatterns:
                if pattern.startNode == start_node_label and pattern.endNode == end_node_label:
                    return pattern
        raise ValueError(
            f"Query pattern not found for start_node: '{start_node_label}' and end_node: '{end_node_label}'")

    def has_query_pattern(self, start_node_label:str, end_node_label:str) -> bool:
        if self.queryPatterns:
            for pattern in self.queryPatterns:
                if pattern.startNode == start_node_label and pattern.endNode == end_node_label:
                    return True
        return False

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
    #trackSources: bool = Field(default=True, description="Set to True unless user specifies otherwise. Whether to track the source of each node and relationship.")

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

    def nodes_only_prompt_str(self, **kwargs) -> str:
        return json.dumps({"nodes": [node.model_dump(**kwargs) for node in self.nodes]}, indent=4)

    def prompt_str(self, **kwargs) -> str:
        return json.dumps(self.query_model_dump(**kwargs), indent=4)

    def query_model_to_yaml(self, **kwargs) -> str:
        """
        Serialize the GraphSchema into a YAML string representation.
        Leverages `model_dump` to generate the dictionary and converts it to YAML.
        """
        # Use PyYAML's dump() to convert the dictionary into YAML
        return yaml.dump(self.query_model_dump(**kwargs), sort_keys=False)  # sort_keys=False keeps the field order consistent

    def get_node_schema_by_label(self, label: str) -> NodeSchema:
        """
        Retrieve a specific node schema by its label.
        :param label: The label of the node schema to retrieve.
        :return: The NodeSchema with the given label.
        :raises ValueError: If no NodeSchema with the given label is found.
        """
        for node in self.nodes:
            if node.label == label:
                return node
        raise ValueError(f"No NodeSchema found with the label '{label}'")

    def export(self, file_path):
        """
        Exports graph schema model to a JSON file.

        Args:
            file_path (str): The path to the file where the schema will be saved.
        """
        # Convert the schema to a dictionary and write it to a JSON file
        with open(file_path, 'w') as file:
            json.dump(self.model_dump(), file, indent=4)

    def get_relationship_schema_by_type(self, rel_type: str) -> RelationshipSchema:
        """
        Retrieve a specific relationship schema by its type.
        :param rel_type: The type of the relationship to retrieve.
        :return: The RelationshipSchema that matches the criteria.
        :raises ValueError: If no matching RelationshipSchema is found.
        """
        # Loop through all relationships in the graph schema to check for matches
        for relationship in self.relationships:
            # Check if the relationship type matches
            if relationship.type == rel_type:
                        return relationship

        # If no match is found, raise an error
        raise ValueError(
            f"No RelationshipSchema found with type '{rel_type}'"
        )

    def get_relationship_schema(self, rel_type: str, start_node_label: str, end_node_label: str) -> RelationshipSchema:
        """
        Retrieve a specific relationship schema by its type and start and end node labels.
        :param rel_type: The type of the relationship to retrieve.
        :param start_node_label: The label of the start node.
        :param end_node_label: The label of the end node.
        :return: The RelationshipSchema that matches the criteria.
        :raises ValueError: If no matching RelationshipSchema is found.
        """
        # Loop through all relationships in the graph schema to check for matches
        for relationship in self.relationships:
            # Check if the relationship type matches
            if relationship.type == rel_type:
                # Check if any of the query patterns match the given start and end node labels
                for pattern in relationship.queryPatterns:
                    if pattern.startNode == start_node_label and pattern.endNode == end_node_label:
                        return relationship

        # If no match is found, raise an error
        raise ValueError(
            f"No RelationshipSchema found with type '{rel_type}' and query pattern "
            f"'{start_node_label}-[{rel_type}]->{end_node_label}'"
        )

    def get_node_properties(self, label:str) -> List[str] :
        """
        Gets the properties names from a node schema, including id name.
        Useful for constructing returns in Cypher queries as it avoids search fields such as embeddings
        """
        node = self.get_node_schema_by_label(label)
        return [node.id.name] + [p.name for p  in node.properties]

    def get_node_search_field_name(self, label:str, prop:str):
        node = self.get_node_schema_by_label(label)
        for search_field in node.searchFields:
            if search_field.calculatedFrom == prop:
                return search_field.name
        raise ValueError(f"No search field found for property {prop} in node {label}")

    def get_all_text_embedding_names(self) -> List[str]:
        res = []
        for node in self.nodes:
            for search_field in node.searchFields:
                if search_field.type == "TEXT_EMBEDDING":
                    res.append(search_field.name)
        return res

    #TODO: Add property filters
    def subset(self, sub_schema: SubSchema) -> "GraphSchema":
        """
        Generates a subset of the graph schema based on a SubSchema object.

        Parameters:
        subschema: SubSchema
            An object encapsulating nodes, patterns, relationships, and a custom description for the subset.

        Returns:
        GraphSchema
            A new GraphSchema instance representing the filtered subset of the graph schema.

        Raises:
        ValueError
            If all inputs in the SubSchema are None.
        """
        # get nodes
        node_schemas = dict()
        if sub_schema.nodes:
            for node in sub_schema.nodes:
                node_schemas[node] = self.get_node_schema_by_label(node).model_copy(deep=True)

        # get relationships filtered by query patterns
        relationship_schemas: Dict[str, RelationshipSchema] = dict()
        if sub_schema.patterns:
            for pattern in sub_schema.patterns:
                relationship_schema = self.get_relationship_schema(pattern[1], pattern[0],
                                                                   pattern[2]).model_copy(deep=True)
                if pattern[1] in relationship_schemas:
                    if not relationship_schemas[pattern[1]].has_query_pattern(pattern[0], pattern[2]):
                        relationship_schemas[pattern[1]].queryPatterns += [relationship_schema.get_query_pattern(pattern[0], pattern[2])]
                        #populate nodes
                        if pattern[0] not in node_schemas:
                            node_schemas[pattern[0]] = self.get_node_schema_by_label(pattern[0]).model_copy(deep=True)
                        if pattern[2] not in node_schemas:
                            node_schemas[pattern[2]] = self.get_node_schema_by_label(pattern[2]).model_copy(deep=True)
                else:
                    relationship_schema.queryPatterns = [relationship_schema.get_query_pattern(pattern[0], pattern[2])]
                    relationship_schemas[pattern[1]] = relationship_schema
                    # populate nodes
                    if pattern[0] not in node_schemas:
                        node_schemas[pattern[0]] = self.get_node_schema_by_label(pattern[0]).model_copy(deep=True)
                    if pattern[2] not in node_schemas:
                        node_schemas[pattern[2]] = self.get_node_schema_by_label(pattern[2]).model_copy(deep=True)

        # get relationships - note that this will pull all query patterns regardless of previously provided patterns
        if sub_schema.relationships:
            for relationship in sub_schema.relationships:
                relationship_schema = self.get_relationship_schema_by_type(relationship).model_copy(deep=True)
                relationship_schemas[relationship] = relationship_schema
                for query_pattern in relationship_schema.queryPatterns:
                    if query_pattern.startNode not in node_schemas:
                        node_schemas[query_pattern.startNode] = self.get_node_schema_by_label(query_pattern.startNode).model_copy(deep=True)
                    if query_pattern.endNode not in node_schemas:
                        node_schemas[query_pattern.endNode] = self.get_node_schema_by_label(query_pattern.endNode).model_copy(deep=True)

        description = sub_schema.description if sub_schema.description else (
                self.description + f"\nSubset to just the following nodes: {list(node_schemas.keys())}, "
                                   f"and relationships: {list(relationship_schemas.keys())}")

        return GraphSchema(description=description, nodes=list(node_schemas.values()), relationships=list(relationship_schemas.values()))
