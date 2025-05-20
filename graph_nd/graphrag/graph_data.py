import uuid
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List, Union
from warnings import warn

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from neo4j import RoutingControl
from pydantic import BaseModel, Field
from tqdm import tqdm

from graph_nd.graphrag.graph_schema import NodeSchema, RelationshipSchema, SearchFieldSchema
from graph_nd.graphrag.source_metadata import SourceType, TransformType, LoadType, prepare_source_metadata


def chunks(xs, n=10_000):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]

#TODO: Currently uses UNIQUE instead of Key for Community.  Consider revizing later.
def create_constraint_if_not_exists(node_schema:NodeSchema, db_client) -> bool:
    """
    Create a unique constraint for the node label and property id if it doesn't exist in the database.
    """

    # check for constraint
    constraint_exists = db_client.execute_query(f"""
                        SHOW CONSTRAINTS YIELD *
                        WHERE type IN ["NODE_KEY", "UNIQUENESS"] 
                            AND entityType="NODE" 
                            AND "{node_schema.id.name}" IN properties
                        RETURN count(*) > 0 AS res
                        """, routing_=RoutingControl.WRITE, result_transformer_= lambda r: r.values()[0][0])

    if not constraint_exists:
        #check for b - tree index
        index_exists = db_client.execute_query("""
                        SHOW INDEXES YIELD *
                        WHERE type IN ["RANGE"] AND entityType="NODE" AND "imdbId" IN properties
                        RETURN count(*) > 0 AS res
                        """, routing_=RoutingControl.WRITE, result_transformer_=lambda r: r.values()[0][0])
        # if b-tree index exists throw warning
        if index_exists:
            warn(
                f"WARNING: A range index exists on `{node_schema.label}` for property `{node_schema.id.name}`, "
                "but no unique constraint is present. This can result in undesirable behavior like merging to duplicate nodes.",
                UserWarning
            )

        else:
            # create constraint
            db_client.execute_query(
                f'CREATE CONSTRAINT unique_{node_schema.label.lower()}_{node_schema.id.name} IF NOT EXISTS FOR (n:{node_schema.label}) REQUIRE n.{node_schema.id.name} IS UNIQUE',
                routing_=RoutingControl.WRITE
            )
    return True

#TODO: Can we remove the `skip` argument?
def make_set_clause(prop_names: List[str], element_name='n', item_name='rec', skip=None):
    if skip is None:
        skip = []
    clause_list = []
    for prop_name in prop_names:
        if prop_name not in skip:
            clause_list.append(f'{element_name}.{prop_name} = {item_name}.{prop_name}')
    return 'SET ' + ', '.join(clause_list) if len(clause_list) > 0 else ''

#TODO: Validate no leading __ in all labels, types, and property names. these shoulde be reserved for internals
def make_source_set_clause(source_id, element_name='n'):
    return f'''
    SET {element_name}.__source_id = coalesce({element_name}.__source_id, [])  + ["{source_id}"]
    '''

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

def batch_embed(df: pd.DataFrame, field_to_embed: str, embedding_field_name: str, embedding_model, chunk_size=100) -> Tuple[pd.DataFrame, int]:
    """
    Adds embeddings to the input Pandas DataFrame, working on rows where a specific field is non-null,
    with progress tracking via tqdm.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to process.
        field_to_embed (str): The column whose values will be embedded.
        embedding_field_name (str): The column name for storing the generated embeddings.
        embedding_model: A model instance compatible with LangChain embeddings (like OpenAIEmbeddings).
        chunk_size (int): The number of records to process in a single batch.

    Returns:
        pd.DataFrame: A filtered DataFrame with the added embeddings column.

    Raises:
        ValueError: If the input data is invalid or the embedding model is missing.
    """
    # Validate inputs
    if field_to_embed not in df.columns:
        raise ValueError(f"'{field_to_embed}' does not exist in the provided DataFrame.")

    # Step 1: Filter the DataFrame to keep rows where the field to embed is not null
    filtered_df = df.dropna(subset=[field_to_embed]).copy()

    # Step 2: Generate embeddings for the filtered rows
    embeddings = []
    texts = filtered_df[field_to_embed].to_list()
    #print("[Embedding] Generating embeddings in chunks...")

    # Use tqdm to show progress during embedding generation
    for chunk in chunks(texts, n=chunk_size): #tqdm(chunks(texts, n=chunk_size), desc="Processing embedding chunks"):
        # Generate embeddings for each chunk and extend the embeddings list
        embeddings.extend(embedding_model.embed_documents(chunk))

    # Step 3: Create a new column in the filtered DataFrame for embeddings
    filtered_df[embedding_field_name] = embeddings

    #print("[Embedding] Process completed successfully.")
    return filtered_df, len(embeddings[0])

def validate_and_create_source_node(source_metadata: Dict[str, Any], db_client):

    # Generate a random UUID for `id` if it doesn't exist or is empty
    if "id" not in source_metadata or not source_metadata["id"]:
        source_metadata["id"] = str(uuid.uuid4())

    #create index if not exists
    db_client.execute_query(
        'CREATE INDEX range___source___id IF NOT EXISTS FOR (n:__Source__) ON n.id',
        routing_=RoutingControl.WRITE
    )

    #create query and execute
    prop_str = ', '.join([f'{prop_name}: $rec.{prop_name}' for prop_name in source_metadata.keys()])
    template = f'''MERGE(n:__Source__ {{{prop_str}}}) SET n.createdAt = datetime.realtime()'''
    db_client.execute_query(template, routing_=RoutingControl.WRITE, rec=source_metadata)
    return source_metadata['id']


class NodeData(BaseModel):
    """
    Data representing graph nodes.  With their provided schema
    """
    node_schema: NodeSchema = Field(description="schema for the nodes")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="records of node properties mapping property names to values.")

    def create_fulltext_index_if_not_exists(self, db_client, field:SearchFieldSchema):
        """
        Create fulltext index for the node label and property if it desn't exist in the database.
        """
        db_client.execute_query(
            f'CREATE FULLTEXT INDEX `{field.indexName}` IF NOT EXISTS FOR (n:`{self.node_schema.label}`) ON EACH [n.`{field.calculatedFrom}`]',
            routing_=RoutingControl.WRITE
        )
        # wait for index to come online
        db_client.execute_query(
            f'CALL db.awaitIndex("{field.indexName}", 300)',
            routing_=RoutingControl.WRITE)

    def create_fulltext_indexes_if_not_exists(self, db_client):
        """
        Create fulltext indexes for the node label and properties if they don't exist in the database.
        """
        for field in self.node_schema.searchFields if self.node_schema.searchFields is not None else []:
            if field.type == 'FULLTEXT':
                self.create_fulltext_index_if_not_exists(db_client, field)

    def create_vector_index_if_not_exists(self, db_client, prop_name:str, dim:int):
        """
        Create vector index for the node label and property if it doesn't exist in the database.
        """
        field  = self.node_schema.get_node_search_field(prop_name, "TEXT_EMBEDDING")
        db_client.execute_query(
            f'''CREATE VECTOR INDEX `{field.indexName}` IF NOT EXISTS FOR (n:`{self.node_schema.label}`) ON n.`{field.name}`
            OPTIONS {{ indexConfig: {{
             `vector.dimensions`: {dim},
             `vector.similarity_function`: 'cosine'
            }}}}
            ''',
            routing_=RoutingControl.WRITE)
        # wait for index to come online
        db_client.execute_query(
            f'CALL db.awaitIndex("{field.indexName}", 300)',
            routing_=RoutingControl.WRITE)

    def make_node_merge_query(self, source_id=None):
        template = f'''UNWIND $recs AS rec\nMERGE(n:{self.node_schema.label} {{{self.node_schema.id.name}: rec.{self.node_schema.id.name}}})'''

        # get property names from records and check for consistency
        prop_names = validate_property_names(self.records)
        template = template + '\n' + make_set_clause(prop_names, skip=[self.node_schema.id.name])
        if source_id:
            template = template + '\n' + make_source_set_clause(source_id)
        return template + '\nRETURN count(n) AS nodeLoadedCount'

    def merge_text_emb(self, db_client, embedding_model, emb_chunk_size=1000, load_chunk_size=1000):
        """
        Merge node embedding data into the database.
        """
        #get fields to embed
        embed_maps = [{'emb': field.name, 'prop': field.calculatedFrom}
                      for field in (self.node_schema.searchFields or [])
                      if field.type == 'TEXT_EMBEDDING']
        #loop through
        if len(embed_maps) > 0:
            df = pd.DataFrame(self.records)
            for embed_map in embed_maps: #tqdm(embed_maps, desc="Embedding Node Properties"):
                if embed_map['prop'] in df.columns:
                    if embedding_model is None:
                        raise ValueError(
                            "Embedding model is required to process text embeddings, but 'embedding_model' is None.")
                    # generate embeddings
                    emb_df, dim = batch_embed(df[[self.node_schema.id.name, embed_map['prop']]], field_to_embed=embed_map['prop'],
                                         embedding_field_name=embed_map['emb'],
                                         embedding_model=embedding_model,
                                         chunk_size=emb_chunk_size)
                    #merge embeddings
                    query = f'''
                    UNWIND $recs AS rec
                    MATCH(n:{self.node_schema.label} {{{self.node_schema.id.name}: rec.{self.node_schema.id.name}}})
                    CALL db.create.setNodeVectorProperty(n, "{embed_map['emb']}", rec.{embed_map['emb']})
                    RETURN count(n) AS nodeVectorLoadedCount
                    '''
                    for recs in chunks(emb_df.to_dict('records'), load_chunk_size):
                        db_client.execute_query(query, routing_=RoutingControl.WRITE, recs=recs)
                    #create index if not exists
                    self.create_vector_index_if_not_exists(db_client, embed_map['prop'], dim)


    def merge(self, db_client, source_metadata: Union[bool, Dict[str, Any]]=True, embedding_model=None, chunk_size=1000, emb_gen_chunk_size=1000,
              emb_load_chunk_size=1000):
        """
        Merge node data into the database.
        """
        default_source_metadata = {
            "id": f"merge_nodes_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4().hex[:8]}",
            "sourceType": SourceType.NODE_LIST.value,
            "transformType": TransformType.UNKNOWN.value,
            "loadType": LoadType.MERGE_NODES.value,
            "name": "node-merge",
        }
        source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
        if source_metadata:
            source_id = validate_and_create_source_node(source_metadata, db_client)
            # make query
            query = self.make_node_merge_query(source_id)
        else:
            # make query
            query = self.make_node_merge_query()

        # set constraint
        create_constraint_if_not_exists(self.node_schema, db_client)

        #execute in chunks
        for recs in chunks(self.records, chunk_size):
            db_client.execute_query(query, routing_=RoutingControl.WRITE, recs=recs)

        #text embedding merge and vector index check/creation
        self.merge_text_emb(db_client, embedding_model, emb_chunk_size=emb_gen_chunk_size, load_chunk_size=emb_load_chunk_size)

        #full text indexes
        self.create_fulltext_indexes_if_not_exists(db_client)



class RelationshipData(BaseModel):
    """
    Data representing graph relationships.  With their provided schema
    """
    rel_schema: RelationshipSchema = Field(description="schema for the relationships")
    start_node_schema: NodeSchema = Field(description="schema for the start node")
    end_node_schema: NodeSchema = Field(description="schema for the end node")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="records of relationship properties and"
                                                                            " start/end node ids. "
                                                                            "records must contain 'start_node_id' and 'end_node_id' properties")

    def make_rel_merge_query(self, source_id=None):
        merge_statement = f'MERGE(s)-[r:{self.rel_schema.type}]->(t)'
        skip_set_props = ['start_node_id','end_node_id']
        if self.rel_schema.id is not None:
            merge_statement = f'MERGE(s)-[r:{self.rel_schema.type} {{{self.rel_schema.id.name}: rec.{self.rel_schema.id.name}}}]->(t)'
            skip_set_props.append(self.rel_schema.id.name)

        template = f'''\tUNWIND $recs AS rec
        MERGE(s:{self.start_node_schema.label} {{{self.start_node_schema.id.name}: rec.start_node_id}})
        MERGE(t:{self.end_node_schema.label} {{{self.end_node_schema.id.name}: rec.end_node_id}})\n\t''' + merge_statement

        # get property names from records and check for consistency
        prop_names = validate_property_names(self.records)
        template = template + '\n\t' + make_set_clause(prop_names, 'r', skip=skip_set_props)
        if source_id:
            template = template + '\n' + make_source_set_clause(source_id, 'r')
        return template + '\n\tRETURN count(r) AS relLoadedCount'

    def merge(self, db_client, source_metadata: Union[bool, Dict[str, Any]]=True, chunk_size=1000):
        """
        Merge relationship data into the database.
        """
        default_source_metadata = {
            "id": f"merge_relationships_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4().hex[:8]}",
            "sourceType": SourceType.RELATIONSHIP_LIST.value,
            "transformType": TransformType.UNKNOWN.value,
            "loadType": LoadType.MERGE_RELATIONSHIPS.value,
            "name": "relationship-merge",
        }
        source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
        if source_metadata:
            source_id = validate_and_create_source_node(source_metadata, db_client)
            # make query
            query = self.make_rel_merge_query(source_id)
        else:
            # make query
            query = self.make_rel_merge_query()

        # set constraints
        create_constraint_if_not_exists(self.start_node_schema, db_client)
        create_constraint_if_not_exists(self.end_node_schema, db_client)

        # execute in chunks
        for recs in chunks(self.records, chunk_size):
            db_client.execute_query(query, routing_=RoutingControl.WRITE, recs=recs)



class GraphData(BaseModel):
    """
    Data representing graph nodes relationships, relationshipDatas should never include nodes not in NodeDatas
    """
    nodeDatas: List[NodeData] = Field(default_factory=list, description="list of NodeData records")
    relationshipDatas: Optional[List[RelationshipData]] = Field(default_factory=list, description="list of RelationshipData records")

    def consolidate_node_datas(self):
        """
        Consolidate NodeDatas such that there is only one NodeData per unique node_schema,
        and append the records of duplicate NodeData objects.
        This is critical for merging graph data efficiently.
        """
        # Use a dictionary to hold deduplicated NodeData objects, keyed by node_schema
        deduplicated = {}

        for node_data in self.nodeDatas:
            # Use node_schema as a key for deduplication
            node_schema_key = node_data.node_schema.label

            if node_schema_key in deduplicated:
                # Append records to the existing NodeData object
                deduplicated[node_schema_key].records.extend(node_data.records)
            else:
                # Add a new entry in deduplicated dictionary
                deduplicated[node_schema_key] = node_data
        # set the values (which are the deduplicated NodeData objects)
        self.nodeDatas = list(deduplicated.values())

    def consolidate_relationship_datas(self):
        """
        consolidate RelationshipDatas such that there is only one RelationshipData per unique combination
        of rel_schema, start_node_schema, and end_node_schema.
        Append the records of duplicate RelationshipData objects.
        This is critical for merging graph data efficiently.
        """
        # Use a dictionary to hold deduplicated RelationshipData objects, keyed by (rel_schema, start_node_schema, end_node_schema)
        deduplicated = {}

        for rel_data in self.relationshipDatas:
            # Create a key based on rel_schema, start_node_schema, and end_node_schema
            relationship_key = (rel_data.rel_schema.type, rel_data.start_node_schema.label, rel_data.end_node_schema.label)

            if relationship_key in deduplicated:
                # Append records to the existing RelationshipData object
                deduplicated[relationship_key].records.extend(rel_data.records)
            else:
                # Add a new entry in deduplicated dictionary
                deduplicated[relationship_key] = rel_data

        # Set the deduplicated RelationshipData back to the relationshipDatas attribute
        self.relationshipDatas = list(deduplicated.values())

    def consolidate(self):
        """
        Consolidates node and relationship data by invoking respective methods to
        process data. Makes merging data significantly faster.
        This method acts as a central point to group data consolidation
        operations into a unified workflow.

        """
        self.consolidate_node_datas()
        self.consolidate_relationship_datas()

    def merge(self, db_client, source_metadata: Union[bool, Dict[str, Any]]=True, embedding_model=None):
        default_source_metadata = {
            "id": f"merge_nodes_and_relationships_at_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4().hex[:8]}",
            "sourceType": SourceType.NODE_AND_RELATIONSHIP_LISTS.value,
            "transformType": TransformType.UNKNOWN.value,
            "loadType": LoadType.MERGE_NODES_AND_RELATIONSHIPS.value,
            "name": "node-and-relationship-merge",
        }
        source_metadata = prepare_source_metadata(source_metadata, default_source_metadata)
        for nodeData in tqdm(self.nodeDatas, desc="Merging Nodes by Label", unit="node"):
            #print(f"Merging {nodeData.node_schema.label} nodes")
            nodeData.merge(db_client, source_metadata, embedding_model=embedding_model)

        for relData in tqdm(self.relationshipDatas, desc="Merging Relationships by Type & Pattern", unit="rel"):
            #print(f"Merging ({relData.start_node_schema.label})-[{relData.rel_schema.type}]->({relData.end_node_schema.label}) relationships")
            relData.merge(db_client, source_metadata)

#TODO: Perhaps every merge call in graphrag should be a single transaction that gets commited once all data is in.
