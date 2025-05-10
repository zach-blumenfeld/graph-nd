from typing import Dict, Optional, Tuple, List
import warnings

import pandas as pd

from graph_nd.graphrag.graph_schema import GraphSchema, NodeSchema, RelationshipSchema, PropertySchema, \
    NEO4J_PROPERTY_TYPES, QueryPattern, SearchFieldSchema


def _is_excluded(name, exclude_prefixes, exclude_exact_matches):
    """Check if a name (label, property, or relationship type) should be excluded."""
    if any(name.startswith(prefix) for prefix in exclude_prefixes):
        return True
    if name in exclude_exact_matches:
        return True
    return False

def _get_properties_by_node_label(db_client):
    res = db_client.execute_query("""
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = 'RELATIONSHIP' AND elementType = 'node'
    RETURN label AS nodeLabel, collect({name:property, type:type}) AS properties
    """, result_transformer_= lambda r: r.data())
    return {r['nodeLabel']: r['properties'] for r in res}

def _get_properties_by_rel_type(db_client):
    res = db_client.execute_query("""
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = 'RELATIONSHIP' AND elementType = 'relationship'
    WITH label AS relType, collect({name:property, type:type}) AS properties
    RETURN relType, properties
    """, result_transformer_= lambda r: r.data())
    return {r['relType']: r['properties'] for r in res}

def _get_validated_node_properties(label, properties, exclude_prefixes, exclude_exact_matches, vector_properties) -> Dict[str, str]:
    validated_properties: Dict[str, str] = dict()
    for property in properties:
        if _is_excluded(property['name'], exclude_prefixes, exclude_exact_matches):
            print(f"INFO: Excluding property `{property['name']}` on node `{label}` due to exclusion rules.")
        elif any(property['name'] in vp['properties'] and label in vp['labels'] for vp in vector_properties):
            continue
            #print(f"INFO: Excluding property `{property['name']}` on node `{label}` for the property schema since it is for a vector index. These properties are leveraged for vector search through search fields as specified by the user.")
        elif property['name'] in validated_properties:
            if property['type'] != validated_properties[property['name']]:
                warnings.warn(f"WARNING: Property `{property['name']} `on node `{label}` has conflicting types. found type `{property['type']}` but using `{validated_properties[property['name']]}` for schema.", UserWarning)
        elif property['type'] not in NEO4J_PROPERTY_TYPES:
            print(f"INFO: Excluding property `{property['name']}` on node `{label}` with type `{property['type']}` as that type is unsupported.  Supported types include `{NEO4J_PROPERTY_TYPES}`")
        else:
            validated_properties[property['name']] = property['type']
    return validated_properties


def _get_validated_rel_properties(rel_type, properties, exclude_prefixes, exclude_exact_matches, vector_properties) -> Dict[str, str]:
    validated_properties: Dict[str, str] = dict()
    for property in properties:
        if _is_excluded(property['name'], exclude_prefixes, exclude_exact_matches):
            print(f"INFO: Excluding property `{property['name']}` on relationship `{rel_type}` due to exclusion rules.")
        elif any(property['name'] in vp['properties'] and rel_type in vp['types'] for vp in vector_properties):
            continue
        elif property['name'] in validated_properties:
            if property['type'] != validated_properties[property['name']]:
                warnings.warn(f"WARNING: Property `{property['name']} `on relationship `{rel_type}` has conflicting types. found type `{property['type']}` but using `{validated_properties[property['name']]}` for schema.", UserWarning)
        elif property['type'] not in NEO4J_PROPERTY_TYPES:
            print(f"INFO: Excluding property `{property['name']}` on  relationship `{rel_type}` with type `{property['type']}` as that type is unsupported.  Supported types include `{NEO4J_PROPERTY_TYPES}`")
        else:
            validated_properties[property['name']] = property['type']
    return validated_properties


def _break_tie_on_node_id_prop_candidates(db_client, label: str, id_candidates: List[str]) -> str:
    """
    Break ties among node ID property candidates based on highest count, shortest name,
    and alphabetical order (if needed).

    Parameters:
        label: The label for the node.
        id_candidates: List of candidate properties for determining the node ID.

    Returns:
        The best property name as the node ID.
    """
    # Step 1: Get counts for each candidate
    count_returns = ', '.join([f'count(DISTINCT n.{p}) AS {p}' for p in id_candidates])
    counts: Dict[str, int] = db_client.execute_query(f"""
            MATCH(n:{label})
            RETURN {count_returns}
            """, result_transformer_= lambda r: r.data()[0])
    # Step 2: Sort candidates by:
    #  - Descending count (-count sorts highest first)
    #  - Shortest length (len(x))
    #  - Alphabetical order (x)
    sorted_candidates = sorted(id_candidates, key=lambda x: (-counts[x], len(x), x))

    # Step 3: Return the first candidate (best match after sorting)
    return sorted_candidates[0]


def _find_node_id_without_constraint(db_client, label: str, valid_property_candidates: List[str]) -> Tuple[str, str]:
    properties = db_client.execute_query(f"""
    SHOW INDEXES YIELD *
    WHERE type='RANGE' AND entityType='NODE' AND labelsOrTypes[0] = '{label}'
    RETURN properties
    """, result_transformer_= lambda r: r.value())
    valid_id_properties = []
    for prop in properties:
        if prop in valid_property_candidates:
            valid_id_properties.append(prop)

    if len(valid_id_properties) == 1:
        print(f"INFO: Choosing to use {valid_id_properties[0]} as the node id for node `{label}` as it has a range index.")
        return valid_id_properties[0], "indexed id property. Not guaranteed to be unique"
    elif len(valid_id_properties) > 1:
        res = _break_tie_on_node_id_prop_candidates(db_client, label, valid_id_properties)
        warnings.warn(f"Multiple valid properties with range index found on node `{label}``: `{valid_id_properties}`. Choosing to use `{res}` as the node id.")
        return res, "indexed id property. Not guaranteed to be unique"
    else:
        res = _break_tie_on_node_id_prop_candidates(db_client, label, valid_property_candidates)
        warnings.warn(f"No valid property with range index found on node `{label}`. Choosing to use `{res}` as the node id.")
        return res, "default id property. This node didn't seem to have a great id property so this was chosen by default. Not guaranteed to be unique and not indexed for fast query performance."


def _find_node_id(db_client, label: str, label_constraints:List[Dict], valid_property_candidates: List[str]) -> Tuple[str, str]:

    label_constraint_candidates = []
    for label_constraint in label_constraints:
        if label_constraint['label'] == label:
            if len(label_constraint['properties']) > 1:
                warnings.warn(f"Composite constraint found for `{label_constraint['properties']}` on node `{label}`. It will be ignored as composite-constraints aren't supported.")
            else:
                label_constraint_candidates.append(label_constraint['properties'][0])
    if len(label_constraint_candidates) == 1:
        return label_constraint_candidates[0], "unique identifier"
    elif len(label_constraint_candidates) > 1:
        res = _break_tie_on_node_id_prop_candidates(db_client, label, label_constraint_candidates)
        warnings.warn(f"Multiple valid non-composite node constraints found on node `{label}`: `{label_constraint_candidates}`. Choosing to use `{res}` as the node id.")
        return res, "unique identifier"
    else:
        warnings.warn(f"No valid non-composite node constraints found on node `{label}`. Falling back to range index lookup.")
        return _find_node_id_without_constraint(db_client, label, valid_property_candidates)


def _get_fulltext_search_fields(label:str, valid_properties:List[str], index_df) -> List[SearchFieldSchema]:
    # filter df to fulltext with label
    filtered_df = index_df[(index_df['type'] == 'FULLTEXT') & (index_df['label'] == label)].copy()
    #if filtered df has no rows return empty list
    if filtered_df.shape[0] == 0:
        return []
    #else more expensive checking
    result: List[SearchFieldSchema] = []
    for prop in valid_properties:
        prop_df = filtered_df[filtered_df['property'] == prop]
        if prop_df.shape[0] > 0:
            for ind, row in prop_df.iterrows():
                if len(row['labels']) > 1:
                    warnings.warn(f"WARNING: The fulltext index {row['indexName']} on labels {row['labels']} was encountered and will be ignored as fulltext indexes on multi-labels and multi-properties aren't supported yet.", UserWarning)
                elif len(row['properties']) > 1:
                    warnings.warn(f"WARNING: The fulltext index {row['indexName']} on properties {row['properties']} was encountered and will be ignored as fulltext indexes on multi-labels and multi-properties aren't supported yet.", UserWarning)
                else:
                    result.append(
                        SearchFieldSchema(
                            description='',
                            name=prop,
                            type="FULLTEXT",
                            calculatedFrom=prop,
                            indexName=row['indexName']
                        )
                    )
    return result

def _validate_vector_index_map(index_df: pd.DataFrame, index_map: Dict[str, str]):
    indices = index_df.loc[index_df['type'] == 'VECTOR', 'indexName'].unique()
    for ind, calc_prop in index_map.items():
        if ind not in indices:
            raise ValueError(f"The Vector index `{ind}` does not exist for any node properties in the database.")

def _get_vector_search_fields(label:str, valid_properties:List[str], index_df:pd.DataFrame, index_map: Dict[str,str]) -> List[SearchFieldSchema]:
    # filter df to fulltext with label
    filtered_df = index_df[(index_df['type'] == 'VECTOR') &
                           (index_df['label'] == label) &
                           (index_df['indexName'].isin(index_map.keys()))].copy()
    #if filtered df has no rows return empty list
    if filtered_df.shape[0] == 0:
        return []
    #else more expensive checking
    result: List[SearchFieldSchema] = []
    for ind, calc_prop in index_map.items():
        ind_df = filtered_df[filtered_df['indexName'] == ind]
        if calc_prop in valid_properties:
            for i, row in ind_df.iterrows():
                if len(row['labels']) > 1:
                    warnings.warn(f"WARNING: The vector index {row['indexName']} on labels {row['labels']} was encountered and will be ignored as vector indexes on multi-labels and multi-properties aren't supported yet.", UserWarning)
                elif len(row['properties']) > 1:
                    warnings.warn(f"WARNING: The vector index {row['indexName']} on properties {row['properties']} was encountered and will be ignored as vector indexes on multi-labels and multi-properties aren't supported yet.", UserWarning)
                else:
                    result.append(
                        SearchFieldSchema(
                            description='',
                            name=row['property'],
                            type="TEXT_EMBEDDING",
                            calculatedFrom=calc_prop,
                            indexName=ind
                        )
                    )
    return result

def _check_text_embedding_index_completeness(index_map: Dict[str,str], search_fields:List[SearchFieldSchema]):
    for ind, calc_prop in index_map.items():
        found = False
        for sf in search_fields:
            if sf.calculatedFrom == calc_prop and sf.indexName == ind and sf.type == "TEXT_EMBEDDING":
                found = True
                break
        if not found:
            raise ValueError(f"The text embedding field for `{{{ind}:{calc_prop}}}` was never found. This is likely due to the `{calc_prop}` not being of valid node property or associated with the same node label as the vector index named `{ind}`.")


def create_node_schemas_from_existing_db(
        db_client,
        exclude_prefixes=("_", " "),
        exclude_exact_matches=None,
        text_embed_index_map: Optional[Dict[str,str]]=None,
) -> List[NodeSchema]:
    """
    Creates a list of `NodeSchema` objects for the nodes existing in the database.

    This function interacts with the database using the provided `db_client` to
    retrieve node labels, properties, constraints, and search index information.
    Then, it constructs schemas for each node that fulfill the given inclusion rules.
    Nodes and their properties can be excluded based on prefixes, exact matches,
    or unsupported data types. Additionally, vector text embedding indexes are
    included in the node schema if specified.

    Parameters:
    db_client: The database client interface used for executing queries and
        retrieving schema information.
    exclude_prefixes: A tuple of strings containing prefixes. Node labels or properties
        starting with any of these prefixes are excluded, defaults to ("_", " ").
    exclude_exact_matches: An optional set of exact node label or property names to
        exclude from the schema, defaults to None if not provided.
    text_embed_index_map: An optional dictionary mapping {text_embedding_index_name: text_property}
        where text_property is a node property that is used to calculate the embedding. This is required to use
        text embedding search fields. If not provided, no text embedding search fields will be included in the schema.
        Defaults to None.

    Returns:
    List[NodeSchema]: A list of `NodeSchema` objects, each representing the schema
        of a node in the database, including its properties, search fields, and
        other relevant metadata.

    Raises:
    UserWarning: Raised in cases where a node is excluded due to lacking properties
        or valid data types.
    """

    # Initialize exclusion settings
    exclude_exact_matches = exclude_exact_matches or set()

    # get labels, properties, constraints, and fulltext and vector indexes
    labels = db_client.execute_query("CALL db.labels()", result_transformer_= lambda r: r.value())
    label_properties = _get_properties_by_node_label(db_client)
    vector_properties = db_client.execute_query("""
                        SHOW INDEXES YIELD *
                        WHERE type='VECTOR' AND entityType='NODE'
                        RETURN labelsOrTypes AS labels, properties
                        """, result_transformer_= lambda r: r.data())
    label_constraints =  db_client.execute_query("""
                        SHOW CONSTRAINTS YIELD *
                        WHERE type IN ["NODE_KEY", "UNIQUENESS"] AND entityType="NODE"
                        RETURN labelsOrTypes[0] AS label, properties
                        """, result_transformer_=lambda r: r.data())
    search_index_df = db_client.execute_query("""
                        SHOW INDEXES YIELD *
                        WHERE entityType="NODE" AND type IN ["FULLTEXT", "VECTOR"]
                        RETURN name AS indexName, type, labelsOrTypes AS labels, properties""",
                                                      result_transformer_ = lambda r: r.to_df())
    search_index_df = (search_index_df
     .join(search_index_df['labels'].explode().rename('label'))
     .join(search_index_df['properties'].explode().rename('property')))

    node_list = []
    all_text_emb_field_schemas = []
    for label in labels:
        # Exclude the node if its label is excluded
        if _is_excluded(label, exclude_prefixes, exclude_exact_matches):
            print(f"INFO: Excluding node label '{label}' due to exclusion rules.")
        # Exclude if no node properties
        elif label not in label_properties:
            warnings.warn(f"Excluding node label '{label}' due to no properties.", UserWarning)
        else:
            # get validated properties
            properties = _get_validated_node_properties(label, label_properties[label], exclude_prefixes, exclude_exact_matches, vector_properties)
            # check for no valid properties and exclude if so
            if not properties:
                warnings.warn(f"Excluding node label '{label}' due to no properties with supported data types.", UserWarning)
            else:
                # get node id
                node_id, node_id_desc = _find_node_id(db_client, label, label_constraints, list(properties.keys()))
                property_schemas = []
                node_id_schema = PropertySchema(description=node_id_desc, name=node_id, type=properties[node_id])
                # create property schemas
                for prop_name, prop_type in properties.items():
                    # TODO: This is a wierd property of graph schemas that should be removed in future refactors
                    ## to optimize the data structure.
                    ## Node id properties need not exist in the property schemas
                    ## Duplicating node id in property schemas results in minor inefficiencies for merging data
                    ## with little impact on performance, and is otherwise harmless.
                    if prop_name != node_id:
                        property_schemas.append(PropertySchema(name=prop_name, type=prop_type, description=""))
                # create search field schemas
                search_field_schemas = _get_fulltext_search_fields(label, list(properties.keys()),
                                                                   search_index_df)
                if text_embed_index_map:
                    _validate_vector_index_map(search_index_df, text_embed_index_map)
                    text_emb_field_schemas = _get_vector_search_fields(label,
                                                                       list(properties.keys()),
                                                                       search_index_df,
                                                                       text_embed_index_map)

                    search_field_schemas += text_emb_field_schemas
                    all_text_emb_field_schemas += text_emb_field_schemas
                # append node schema
                node_list.append(NodeSchema(id=node_id_schema,
                                            label=label,
                                            properties=property_schemas,
                                            searchFields=search_field_schemas,
                                            description=""))
    if text_embed_index_map:
        _check_text_embedding_index_completeness(text_embed_index_map, all_text_emb_field_schemas)
    return node_list

def create_relationship_schemas_from_existing_db(db_client,
                                                 node_labels:List[str],
                                                 exclude_prefixes=("_", " "),
                                                 exclude_exact_matches=None,
                                                 parallel_rel_ids:Optional[Dict[str,str]]=None) -> List[RelationshipSchema]:
    """
    Creates relationship schemas by analyzing an existing database structure.

    This function generates a list of relationship schemas (`RelationshipSchema`) derived from
    the existing database, based on defined node labels, exclusion rules, and optionally
    specified parallel relationship IDs. The function inspects relationship patterns,
    validates their inclusion against exclusion rules, retrieves relevant properties, and constructs
    metadata for each relationship type.

    Parameters:
        db_client: The database client used to execute database queries.
        node_labels (List[str]): A list of node labels to consider when filtering relationships.
            Relationships must exist between these node labels to be included.
        exclude_prefixes: A tuple of strings containing prefixes. Relationship types or properties
            starting with any of these prefixes are excluded, defaults to ("_", " ").
        exclude_exact_matches: An optional set of exact relationship type, or property names to
            exclude from the schema, defaults to None if not provided.
        parallel_rel_ids (Optional[Dict[str, str]], optional): A dictionary mapping relationship
            types to their parallel relationship ID property names: `{rel_type: property_name}`. This is only required if the
            user wishes to ingest more data while maintaining parallel relationships for specific node types
            (more than one instance of a relationship type existing between the same start and end nodes). Defaults to None.

    Returns:
        List[RelationshipSchema]: A list of constructed relationship schema objects.

    Raises:
        ValueError: If a relationship type defined in `parallel_rel_ids` is missing or refers to
        an unsupported field in the database or schema.
        ValueError: If a provided parallel ID property does not exist or its type is unsupported.
    """
    # Initialize exclusion settings
    exclude_exact_matches = exclude_exact_matches or set()

    # get rel_patterns & properties
    rel_patterns = db_client.execute_query("""
                    CALL apoc.meta.data()
                    YIELD label, other, elementType, type, property
                    WHERE type = 'RELATIONSHIP' AND elementType = 'node'
                    UNWIND other AS other_node
                    WITH *
                    RETURN {start: label, type: property, end: toString(other_node)} AS output
                    """, result_transformer_= lambda r: r.value())
    rel_properties = _get_properties_by_rel_type(db_client)
    vector_properties = db_client.execute_query("""
                        SHOW INDEXES YIELD *
                        WHERE type='VECTOR' AND entityType='RELATIONSHIP'
                        RETURN labelsOrTypes AS types, properties
                        """, result_transformer_= lambda r: r.data())

    #consolidate relationship patterns under type
    rel_dict = dict()
    for rel_pattern in rel_patterns:
        # silently exclude the rel pattern if start or end are not in node_labels
        if (rel_pattern['start'] not in node_labels) or (rel_pattern['end'] not in node_labels):
            #print(f"INFO: Excluding relationship pattern `{rel_pattern}` due to start or end node not being in the provided node labels: `{node_labels}`.")
            continue
        elif rel_pattern['type'] in rel_dict:
            rel_dict[rel_pattern['type']].append(rel_pattern)
        else:
            rel_dict[rel_pattern['type']] = [rel_pattern]

    #validate parallel rel id
    if parallel_rel_ids is None:
        parallel_rel_ids = {}
    else:
        for rel_type, rel_id in parallel_rel_ids.items():
            if rel_type not in rel_dict:
                raise ValueError(
                    f"The provided relationship type `{rel_type}` from parallel_rel_ids doesn't exist."
                )

    # logic by relationship type
    rel_schemas = []
    for rel_type, rel_patterns in rel_dict.items():
        # Exclude the rel pattern if type hits exclusion rule
        if _is_excluded(rel_type, exclude_prefixes, exclude_exact_matches):
            print(f"INFO: Excluding relationship type `{rel_type}` due to exclusion rules.")
        else:
            # get valid properties
            if rel_type in rel_properties:
                properties = _get_validated_rel_properties(rel_type, rel_properties[rel_type], exclude_prefixes, exclude_exact_matches, vector_properties)
            else:
                properties={}
            # get parallel rel id if available
            if rel_type in parallel_rel_ids:
                parallel_rel_id = parallel_rel_ids[rel_type]
                if not parallel_rel_id in properties:
                    raise ValueError(
                        f"The provided parallel id property `{parallel_rel_id}` for relationship type `{rel_type}` is missing or does not have a supported data type."
                    )
                else:
                    parallel_rel_id_schema = PropertySchema(name=parallel_rel_id, type=properties.pop(parallel_rel_id), description="")
            else:
                parallel_rel_id_schema = None
            # create property schema
            property_schemas = []
            for prop_name, prop_type in properties.items():
                property_schemas.append(PropertySchema(name=prop_name, type=prop_type, description=""))
            query_patterns = []
            for rel_pattern in rel_patterns:
                query_patterns.append(QueryPattern(startNode=rel_pattern['start'], endNode=rel_pattern['end'], description=''))
            rel_schemas.append(RelationshipSchema(type=rel_type, properties=property_schemas, queryPatterns=query_patterns, id=parallel_rel_id_schema, description=""))
    return rel_schemas

def create_graph_schema_from_existing_db(db_client,
                                         exclude_prefixes=("_", " "),
                                         exclude_exact_matches=None,
                                         text_embed_index_map: Optional[Dict[str, str]] = None,
                                         parallel_rel_ids:Optional[Dict[str,str]]=None,
                                         description=None) -> GraphSchema:
    """
        Creates a graph schema from an existing database.

        This function analyzes the existing database to generate node schemas and
        relationship schemas, combining them into a unified graph schema. Users can
        specify customization options such as excluding certain prefixes or specific
        exact matches for attributes, and optionally map text embeddings or parallel
        relationship IDs. A description for the schema can also be provided.

        Args:
            db_client: The database client used to interact with the database.
            exclude_prefixes: A tuple of strings containing prefixes. Node labels, relationship types, or properties
                starting with any of these prefixes are excluded, defaults to ("_", " ").
            exclude_exact_matches: An optional set of exact node labels, relationship types, or property names to
                exclude from the schema, defaults to None if not provided.
            text_embed_index_map: An optional dictionary mapping {text_embedding_index_name: text_property}
                where text_property is a node property that is used to calculate the embedding. This is required to use
                text embedding search fields. If not provided, no text embedding search fields will be included in the schema.
                Defaults to None.
            parallel_rel_ids (Optional[Dict[str, str]], optional): A dictionary mapping relationship
                types to their parallel relationship ID property names: `{rel_type: property_name}`. This is only required if the
                user wishes to ingest more data while maintaining parallel relationships for specific node types
                (more than one instance of a relationship type existing between the same start and end nodes). Defaults to None.
            description: Optional description of the generated graph schema. Exposed to LLM when accessing the graph through GraqphRAG.

        Returns:
            GraphSchema: The graph schema constructed, comprising node schemas
            and relationship schemas extracted from the existing database.
    """
    nodes = create_node_schemas_from_existing_db(db_client, exclude_prefixes, exclude_exact_matches, text_embed_index_map)
    relationships = create_relationship_schemas_from_existing_db(db_client, [n.label for n in nodes], exclude_prefixes, exclude_exact_matches, parallel_rel_ids)
    return GraphSchema(nodes=nodes, relationships=relationships, description= description if description else "graph schema found in existing database")