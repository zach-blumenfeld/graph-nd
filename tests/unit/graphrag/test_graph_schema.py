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

def test_property_schema_allowed_values_string_valid():
    """Test that a STRING property with valid allowed_values passes validation"""
    prop = PropertySchema(
        name="status",
        type="STRING",
        description="Status of the entity",
        allowed_values=["pending", "active", "completed"]
    )
    # No exception should be raised
    assert prop.allowed_values == ["pending", "active", "completed"]


def test_property_schema_allowed_values_string_invalid():
    """Test that a STRING property with invalid allowed_values fails validation"""
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError) as exc_info:
        PropertySchema(
            name="status",
            type="STRING",
            description="Status of the entity",
            allowed_values=["pending", 123, "completed"]  # 123 is not a string
        )

    # Check that the error message includes the property name
    assert "Property 'status'" in str(exc_info.value)
    assert "must be strings" in str(exc_info.value)


def test_property_schema_allowed_values_integer_valid():
    """Test that an INTEGER property with valid allowed_values passes validation"""
    prop = PropertySchema(
        name="priority",
        type="INTEGER",
        description="Priority level",
        allowed_values=[1, 2, 3, 5, 8]
    )
    # No exception should be raised
    assert prop.allowed_values == [1, 2, 3, 5, 8]


def test_property_schema_allowed_values_integer_invalid():
    """Test that an INTEGER property with invalid allowed_values fails validation"""
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError) as exc_info:
        PropertySchema(
            name="priority",
            type="INTEGER",
            description="Priority level",
            allowed_values=[1, "medium", 3]  # "medium" is not an integer
        )

    # Check that the error message includes the property name
    assert "Property 'priority'" in str(exc_info.value)
    assert "must be integers" in str(exc_info.value)


def test_property_schema_allowed_values_unsupported_type():
    """Test that allowed_values is rejected for unsupported property types"""
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError) as exc_info:
        PropertySchema(
            name="is_active",
            type="BOOLEAN",
            description="Whether the entity is active",
            allowed_values=[True, False]
        )

    # Check that the error message is clear about supported types
    assert "Property 'is_active'" in str(exc_info.value)
    assert "only supported for STRING and INTEGER" in str(exc_info.value)


def test_property_schema_allowed_values_none():
    """Test that allowed_values can be None for any property type"""
    # Test with STRING
    prop_string = PropertySchema(
        name="name",
        type="STRING",
        description="Name of the entity",
        #allowed_values=None
    )
    assert prop_string.allowed_values is None

    # Test with INTEGER
    prop_int = PropertySchema(
        name="count",
        type="INTEGER",
        description="Count value",
        #allowed_values=None
    )
    assert prop_int.allowed_values is None

    # Test with BOOLEAN
    prop_bool = PropertySchema(
        name="is_active",
        type="BOOLEAN",
        description="Activity status",
        #allowed_values=None
    )
    assert prop_bool.allowed_values is None
