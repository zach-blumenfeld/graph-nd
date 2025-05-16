from typing import Dict, Optional, Tuple, List
import warnings
from graph_nd.graphrag.graph_schema import GraphSchema, NodeSchema, RelationshipSchema, PropertySchema, \
    NEO4J_PROPERTY_TYPES, QueryPattern  # assume imports for schema classes

# Helper functions
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
    WITH label AS relType, collect({property:property, type:type}) AS properties
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

def create_node_schemas_from_existing_db(
        db_client,
        exclude_prefixes=("_", " "),
        exclude_exact_matches=None,
        text_embedding_fields=None,
) -> List[NodeSchema]:

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
    node_list = []
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
                # get fulltext embeddings
                # create property schema
                for prop_name, prop_type in properties.items():
                    if prop_name != node_id:
                        property_schemas.append(PropertySchema(name=prop_name, type=prop_type, description=""))
                # append node schema
                node_list.append(NodeSchema(id=node_id_schema, label=label, properties=property_schemas, searchFields=[], description=""))
    return node_list

def create_relationship_schemas_from_existing_db(db_client,
                                                 node_labels:List[str],
                                                 exclude_prefixes=("_", " "),
                                                 exclude_exact_matches=None,
                                                 parallel_rel_ids:Optional[Dict[str,str]]=None) -> List[RelationshipSchema]:

    if parallel_rel_ids is None:
        parallel_rel_ids = {}
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
                                         text_embedding_fields=None,
                                         parallel_rel_ids:Optional[Dict[str,str]]=None,
                                         description=None) -> GraphSchema:
    nodes = create_node_schemas_from_existing_db(db_client, exclude_prefixes, exclude_exact_matches, text_embedding_fields)
    relationships = create_relationship_schemas_from_existing_db(db_client, [n.label for n in nodes], exclude_prefixes, exclude_exact_matches, parallel_rel_ids)
    return GraphSchema(nodes=nodes, relationships=relationships, description= description if description else "graph schema found in existing database")