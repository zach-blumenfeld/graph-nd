from graph_nd.graphrag.graph_schema import GraphSchema, NodeSchema, RelationshipSchema, PropertySchema, SearchFieldSchema, QueryPattern

import pytest 

@pytest.fixture
def property_a() -> PropertySchema:
    return PropertySchema(name="a", type="STRING")

@pytest.fixture
def property_b() -> PropertySchema:
    return PropertySchema(name="b", type="INTEGER")

@pytest.fixture
def property_c() -> PropertySchema:
    return PropertySchema(name="c", type="BOOLEAN")

@pytest.fixture
def property_d() -> PropertySchema:
    return PropertySchema(name="d", type="FLOAT")

@pytest.fixture
def node_a(property_a: PropertySchema) -> NodeSchema:
    return NodeSchema(label="A", id=property_a, properties=[property_a])

@pytest.fixture
def node_b(property_b: PropertySchema, property_c: PropertySchema) -> NodeSchema:
    return NodeSchema(label="B", id=property_b, properties=[property_b, property_c])

@pytest.fixture
def query_pattern_ab() -> QueryPattern:
    return QueryPattern(startNode="A", endNode="B")

@pytest.fixture
def query_pattern_cb() -> QueryPattern:
    return QueryPattern(startNode="C", endNode="B")

@pytest.fixture
def query_pattern_bc() -> QueryPattern:
    return QueryPattern(startNode="B", endNode="C")

@pytest.fixture
def relationship_ab(node_a: NodeSchema, node_b: NodeSchema, property_d: PropertySchema, query_pattern_ab: QueryPattern) -> RelationshipSchema:
    return RelationshipSchema(type="AB", startNode=node_a, endNode=node_b, properties=[property_d], queryPatterns=[query_pattern_ab])




def test_property_schema_validation_valid_neo4j_type():
    PropertySchema(name="a", type="STRING")

def test_property_schema_validation_invalid_neo4j_type():
    with pytest.raises(ValueError) as e:
        PropertySchema(name="a", type="str")
    assert "Invalid property `type`. Must be one of:" in str(e.value)



def test_node_schema_validation_valid(property_a: PropertySchema):
    NodeSchema(label="A", id=property_a, properties=[property_a])

def test_node_schema_validation_invalid_no_id():
    with pytest.raises(ValueError):
        NodeSchema(label="A", properties=[property_a])




def test_node_schema_validation_invalid_search_fields_not_in_properties(property_a: PropertySchema):
    search_field = SearchFieldSchema(name="embedding", type="TEXT_EMBEDDING", calculatedFrom="b")
    with pytest.raises(ValueError) as e:
        NodeSchema(label="A", id=property_a, properties=[property_a], searchFields=[search_field])
    assert "`searchFields` must only contain" in str(e.value)

def test_relationship_schema_validation_valid_with_properties(property_d: PropertySchema, query_pattern_ab: QueryPattern):
    relationship = RelationshipSchema(type="AB", properties=[property_d], queryPatterns=[query_pattern_ab])
    assert relationship.properties == [property_d]

def test_relationship_schema_validation_valid_no_properties(query_pattern_ab: QueryPattern):
    relationship = RelationshipSchema(type="AB", queryPatterns=[query_pattern_ab])
    assert relationship.properties == []

def test_relationship_schema_validation_valid_id_with_no_properties(property_d: PropertySchema, query_pattern_ab: QueryPattern):
    relationship = RelationshipSchema(type="AB", id=property_d, queryPatterns=[query_pattern_ab])

def test_relationship_schema_validation_invalid_no_query_patterns():
    with pytest.raises(ValueError) as e:
        RelationshipSchema(type="AB")
    

def test_graph_schema_validation(node_a: NodeSchema, node_b: NodeSchema, relationship_ab: RelationshipSchema):
    GraphSchema(nodes=[node_a, node_b], relationships=[relationship_ab])

def test_graph_schema_validation_invalid_start_node_not_in_relationship_pattern(node_a: NodeSchema, node_b: NodeSchema, query_pattern_cb: QueryPattern):
    with pytest.raises(ValueError) as e:
        GraphSchema(nodes=[node_a, node_b], relationships=[RelationshipSchema(type="CB", queryPatterns=[query_pattern_cb])])
    assert "Relationship CB has `queryPattern` with `startNode` label: C not found in `nodes` field." in str(e.value)

def test_graph_schema_validation_invalid_end_node_not_in_relationship_pattern(node_a: NodeSchema, node_b: NodeSchema, query_pattern_bc: QueryPattern):
    with pytest.raises(ValueError) as e:
        GraphSchema(nodes=[node_a, node_b], relationships=[RelationshipSchema(type="BC", queryPatterns=[query_pattern_bc])])
    assert "Relationship BC has `queryPattern` with `endNode` label: C not found in `nodes` field." in str(e.value)