from graph_nd.graphrag.graph_schema import GraphSchema, NodeSchema, RelationshipSchema, PropertySchema, SearchFieldSchema

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
def node_a(property_a: PropertySchema) -> NodeSchema:
    return NodeSchema(label="A", id=property_a, properties=[property_a])

@pytest.fixture
def node_b(property_b: PropertySchema, property_c: PropertySchema) -> NodeSchema:
    return NodeSchema(label="B", id=property_b, properties=[property_b, property_c])







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


def test_node_schema_validation_invalid_no_properties():
    with pytest.raises(ValueError) as e:
        NodeSchema(label="A", properties=[])
    assert "`properties` field must contain at least one property" in str(e.value)

def test_node_schema_validation_valid_id_property_not_in_properties(property_a: PropertySchema, property_b: PropertySchema):
    node = NodeSchema(label="A", id=property_a, properties=[property_b])
    assert node.properties == [property_b, property_a]

def test_node_schema_validation_invalid_search_fields_not_in_properties(property_a: PropertySchema):
    search_field = SearchFieldSchema(name="embedding", type="TEXT_EMBEDDING", calculatedFrom="b")
    with pytest.raises(ValueError) as e:
        NodeSchema(label="A", id=property_a, properties=[property_a], searchFields=[search_field])
    assert "`searchFields` must only contain" in str(e.value)

def test_relationship_schema_validation_valid_with_properties():
    pass

def test_relationship_schema_validation_valid_no_properties():
    pass

def test_relationship_schema_validation_invalid_id_with_no_properties():
    pass

def test_relationship_schema_validation_invalid_no_query_patterns():
    pass

def test_graph_schema_validation():
    pass

def test_graph_schema_validation_invalid_start_node_not_in_nodes():
    pass

def test_graph_schema_validation_invalid_end_node_not_in_nodes():
    pass