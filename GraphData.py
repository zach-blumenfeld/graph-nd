from typing import Any, Dict, Tuple, Optional, List

from neo4j import RoutingControl
from pydantic import BaseModel, Field

from GraphSchema import NodeSchema, RelationshipSchema


def chunks(xs, n=10_000):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]

def make_set_clause(prop_names: List[str], element_name='n', item_name='rec', skip=None):
    if skip is None:
        skip = []
    clause_list = []
    for prop_name in prop_names:
        if prop_name not in skip:
            clause_list.append(f'{element_name}.{prop_name} = {item_name}.{prop_name}')
    return 'SET ' + ', '.join(clause_list) if len(clause_list) > 0 else ''


def validate_property_names(records: List[Dict[str, Any]]) -> List[str]:
    """
    Validate that all records have consistent property names (keys in the dictionaries).
    Returns the list of property names if consistent, raises ValueError otherwise.
    """
    if not records:
        raise ValueError("No records provided, unable to validate property names.")

    # Extract property names from each record
    property_name_sets = {frozenset(record.keys()) for record in records}

    # Check if all records have the same property names
    if len(property_name_sets) > 1:
        raise ValueError(
            "Inconsistent property names found in records. "
            f"Differences: {property_name_sets}"
        )

    # Return the list of property names (convert from the only set in `property_name_sets`)
    return list(property_name_sets.pop())

class NodeData(BaseModel):
    node_schema: NodeSchema = Field(description="schema for the nodes")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="records of node properties")

    #TODO: Currently uses UNIQUE instead of Key for Community.  Consider revizing later.
    def create_constraint_if_not_exists(self, db_client):
        """
        Create a unique constraint for the node label and property id if it doesn't exist in the database.
        """
        db_client.execute_query(
            f'CREATE CONSTRAINT unique_{self.node_schema.label.lower()}_{self.node_schema.id.name} IF NOT EXISTS FOR (n:{self.node_schema.label}) REQUIRE n.{self.node_schema.id.name} IS UNIQUE',
            routing_=RoutingControl.WRITE
        )

    def make_node_merge_query(self):
        template = f'''UNWIND $recs AS rec\nMERGE(n:{self.node_schema.label} {{{self.node_schema.id.name}: rec.{self.node_schema.id.name}}})'''

        # get property names from records and check for consistency
        prop_names = validate_property_names(self.records)
        template = template + '\n' + make_set_clause(prop_names, skip=[self.node_schema.id.name])
        return template + '\nRETURN count(n) AS nodeLoadedCount'

    def merge(self, db_client, chunk_size=1000):
        """
        Merge node data into the database.
        """
        # set constraint
        self.create_constraint_if_not_exists(db_client)

        #make query
        query = self.make_node_merge_query()

        #execute in chunks
        for recs in chunks(self.records, chunk_size):
            db_client.execute_query(query, routing_=RoutingControl.WRITE, recs=recs)


class RelationshipData(BaseModel):
    rel_schema: RelationshipSchema = Field(description="schema for the relationships")
    start_node_schema: NodeSchema = Field(description="schema for the start node")
    end_node_schema: NodeSchema = Field(description="schema for the end node")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="records of relationship properties and"
                                                                            " start/end node ids. "
                                                                            "records must contain 'start_node_id' and 'end_node_id' properties")

    def make_rel_merge_query(self):
        merge_statement = f'MERGE(s)-[r:{self.rel_schema.type}]->(t)'
        skip_set_props = ['start_node_id','end_node_id']
        if self.rel_schema.id is not None:
            merge_statement = f'MERGE(s)-[r:{self.rel_schema.type} {{{self.rel_schema.id.name}: rec.{self.rel_schema.type}}}]->(t)'
            skip_set_props.append(self.rel_schema.id.name)

        template = f'''\tUNWIND $recs AS rec
        MATCH(s:{self.start_node_schema.label} {{{self.start_node_schema.id.name}: rec.start_node_id}})
        MATCH(t:{self.end_node_schema.label} {{{self.end_node_schema.id.name}: rec.end_node_id}})\n\t''' + merge_statement

        # get property names from records and check for consistency
        prop_names = validate_property_names(self.records)
        template = template + '\n\t' + make_set_clause(prop_names, 'r', skip=skip_set_props)
        return template + '\n\tRETURN count(r) AS relLoadedCount'

    def merge(self, db_client, chunk_size=1000):
        """
        Merge relationship data into the database.
        """
        # make query
        query = self.make_rel_merge_query()

        # execute in chunks
        for recs in chunks(self.records, chunk_size):
            db_client.execute_query(query, routing_=RoutingControl.WRITE, recs=recs)

class GraphData(BaseModel):
    nodeDatas: List[NodeData] = Field(default_factory=list, description="list of NodeData records")
    relationshipDatas: List[RelationshipData] = Field(default_factory=list, description="list of RelationshipData records")

    def merge(self, db_client):
        for nodeData in self.nodeDatas:
            print(f"Merging {nodeData.node_schema.label} nodes")
            nodeData.merge(db_client)

        for relData in self.relationshipDatas:
            print(f"Merging ({relData.start_node_schema.label})-[{relData.rel_schema.type}]->({relData.end_node_schema.label}) relationships")
            relData.merge(db_client)

