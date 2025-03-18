from typing import Any, Dict, Tuple, Optional, List
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from neo4j import RoutingControl
from pydantic import BaseModel, Field
from tqdm import tqdm

from graph_schema import NodeSchema, RelationshipSchema


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




class NodeData(BaseModel):
    """
    Data representing graph nodes.  With their provided schema
    """
    node_schema: NodeSchema = Field(description="schema for the nodes")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="records of node properties mapping property names to values.")

    #TODO: Currently uses UNIQUE instead of Key for Community.  Consider revizing later.
    def create_constraint_if_not_exists(self, db_client):
        """
        Create a unique constraint for the node label and property id if it doesn't exist in the database.
        """
        db_client.execute_query(
            f'CREATE CONSTRAINT unique_{self.node_schema.label.lower()}_{self.node_schema.id.name} IF NOT EXISTS FOR (n:{self.node_schema.label}) REQUIRE n.{self.node_schema.id.name} IS UNIQUE',
            routing_=RoutingControl.WRITE
        )

    def create_fulltext_index_if_not_exists(self, db_client, prop_name):
        """
        Create fulltext index for the node label and property if it desn't exist in the database.
        """
        db_client.execute_query(
            f'CREATE FULLTEXT INDEX fulltext_{self.node_schema.label.lower()}_{prop_name} IF NOT EXISTS FOR (n:{self.node_schema.label}) ON EACH [n.{prop_name}]',
            routing_=RoutingControl.WRITE
        )
        # wait for index to come online
        db_client.execute_query(
            f'CALL db.awaitIndex("fulltext_{self.node_schema.label.lower()}_{prop_name}", 300)',
            routing_=RoutingControl.WRITE)

    def create_fulltext_indexes_if_not_exists(self, db_client):
        """
        Create fulltext indexes for the node label and properties if they don't exist in the database.
        """
        for field in self.node_schema.searchFields if self.node_schema.searchFields is not None else []:
            if field.type == 'FULLTEXT':
                self.create_fulltext_index_if_not_exists(db_client, field.calculatedFrom)

    def create_vector_index_if_not_exists(self, db_client, prop_name, dim):
        """
        Create vector index for the node label and property if it doesn't exist in the database.
        """
        db_client.execute_query(
            f'''CREATE VECTOR INDEX vector_{self.node_schema.label.lower()}_{prop_name} IF NOT EXISTS FOR (n:{self.node_schema.label}) ON n.{prop_name}
            OPTIONS {{ indexConfig: {{
             `vector.dimensions`: {dim},
             `vector.similarity_function`: 'cosine'
            }}}}
            ''',
            routing_=RoutingControl.WRITE)
        # wait for index to come online
        db_client.execute_query(
            f'CALL db.awaitIndex("vector_{self.node_schema.label.lower()}_{prop_name}", 300)',
            routing_=RoutingControl.WRITE)

    def create_text_vector_indexes_if_not_exists(self, db_client, dim=1536):
        """
        Create vector indexes for the node label and properties if they don't exist in the database.
        """
        for field in self.node_schema.searchFields if self.node_schema.searchFields is not None else []:
            if field.type == 'TEXT_EMBEDDING':
                self.create_vector_index_if_not_exists(db_client, field.calculatedFrom, dim)

    def make_node_merge_query(self):
        template = f'''UNWIND $recs AS rec\nMERGE(n:{self.node_schema.label} {{{self.node_schema.id.name}: rec.{self.node_schema.id.name}}})'''

        # get property names from records and check for consistency
        prop_names = validate_property_names(self.records)
        template = template + '\n' + make_set_clause(prop_names, skip=[self.node_schema.id.name])
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
                    #create index inf not exists
                    self.create_vector_index_if_not_exists(db_client, embed_map['emb'], dim)


    def merge(self, db_client, embedding_model=None, chunk_size=1000, emb_gen_chunk_size=1000,
              emb_load_chunk_size=1000):
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

    def make_rel_merge_query(self):
        merge_statement = f'MERGE(s)-[r:{self.rel_schema.type}]->(t)'
        skip_set_props = ['start_node_id','end_node_id']
        if self.rel_schema.id is not None:
            merge_statement = f'MERGE(s)-[r:{self.rel_schema.type} {{{self.rel_schema.id.name}: rec.{self.rel_schema.type}}}]->(t)'
            skip_set_props.append(self.rel_schema.id.name)

        template = f'''\tUNWIND $recs AS rec
        MERGE(s:{self.start_node_schema.label} {{{self.start_node_schema.id.name}: rec.start_node_id}})
        MERGE(t:{self.end_node_schema.label} {{{self.end_node_schema.id.name}: rec.end_node_id}})\n\t''' + merge_statement

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
    """
    Data representing graph nodes relationships, relationshipDatas should never include nodes not in NodeDatas
    """
    nodeDatas: List[NodeData] = Field(default_factory=list, description="list of NodeData records")
    relationshipDatas: Optional[List[RelationshipData]] = Field(default_factory=list, description="list of RelationshipData records")

    def merge(self, db_client, embedding_model=None):
        for nodeData in self.nodeDatas:
            #print(f"Merging {nodeData.node_schema.label} nodes")
            nodeData.merge(db_client, embedding_model=embedding_model)

        for relData in self.relationshipDatas:
            #print(f"Merging ({relData.start_node_schema.label})-[{relData.rel_schema.type}]->({relData.end_node_schema.label}) relationships")
            relData.merge(db_client)

