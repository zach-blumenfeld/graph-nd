from langchain_core.prompts import PromptTemplate

SCHEMA_FROM_DESC_TEMPLATE = PromptTemplate.from_template('''
Generate a graphSchema from the below description.
For any string properties that may require semantic search, include an additional textEmbedding search field,
format the name as <originalPropertyName>_textembedding, and add a description to to guide semantic search:

{context}
''')

SCHEMA_FROM_SAMPLE_TEMPLATE = PromptTemplate.from_template('''
Generate a graphSchema from the below sample of domain data.
For any string properties that may require semantic search, include an additional textEmbedding search field,
format the name as <originalPropertyName>_textembedding, and add a description to to guide semantic search:

{context}
''')

SCHEMA_FROM_DICT_TEMPLATE = PromptTemplate.from_template('''
Craft a graphSchema from the following json-like definition.  
Stay true to this definition when converting to graphSchema.  don't violate it.
Do not under any circumstances skip required node id fields when creating a graphSchema.
If the json-like definition does not seem to include an id field for a node look to see if any of the existing fields could be interpreted as ids. 
If they cannot, add your own id field.
For any string properties that may require semantic search, include an additional textEmbedding search field,
format the name as <originalPropertyName>_textembedding, and add a description to to guide semantic search:

{context}
''')

TABLE_TYPE_TEMPLATE = PromptTemplate.from_template('''
Please tell us the type of table given the tablePreview and graphSchema it maps to.
There are two types of tables:
    - SINGLE_NODE: a table that maps to a single entity (i.e. node) from the graphSchema
    - RELATIONSHIPS: a table that maps to one or more relationships between nodes in the graphSchema 

## tablePreview
Table Name: {tableName}
{tablePreview}

## graphSchema
{graphSchema}
''')

NODE_MAPPING_TEMPLATE = PromptTemplate.from_template('''
The tablePreview represents a table for an entity that we need to map to a node in the below graphSchema.  
Please identify the node and provide this mapping.

## tablePreview
Table Name: {tableName}
{tablePreview}

## graphSchema
{graphSchema}
''')

RELATIONSHIPS_MAPPING_TEMPLATE = PromptTemplate.from_template('''
The tablePreview represents a table for 1 or more relationships between entities that we need to map to the below graphSchema.  
Please identify the relationships and provide this mapping.

## tablePreview
Table Name: {tableName}
{tablePreview}

## graphSchema
{graphSchema}
''')

TEXT_EXTRACTION_TEMPLATE = PromptTemplate.from_template('''
Extract nodes, relationships and their properties where applicable from the textChunk according to the graphSchema. 
This data will be merged into a knowledge graph to support downstream search and analytics. As such:
- You must strictly adhere to the schema
- to find data to populate node ids, use the graph schema to find the node id names in the text.
- every relationship requires start and end ids, do not provide relationships with start and end ids that don't correspond to existing nodes you extracted
- You may find that not all nodes or relationship types are present in the graph schema.  That is okay. 

## TextChunk From File {fileName}
{text}

## GraphSchema
{graphSchema}
''')

QUERY_TEMPLATE = PromptTemplate.from_template("""
Task: Generate a Cypher statement for traversing a Neo4j graph database from a user input. 
- Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
- Do not use any properties or relationships not included in the graphSchema.
- Use the queryPatterns in the graphSchema to guide your query constructions.
- Ignore the searchFields in the graphSchema.

## graphSchema
{graphSchema}

## User Input
{queryInstructions}


Cypher query:
""")


AGG_QUERY_TEMPLATE = PromptTemplate.from_template("""
Task: Generate a Cypher statement for aggregating information in a Neo4j graph database from a user input. 
- Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
- Do not use any properties or relationships not included in the graphSchema.
- Use the queryPatterns in the graphSchema to guide your query constructions.
- Ignore the searchFields in the graphSchema.

## graphSchema
{graphSchema}

## User Input
{queryInstructions}


Cypher query:
""")

AGENT_SYSTEM_TEMPLATE = PromptTemplate.from_template('''
You are an AI assistant that helps users with their inquiries.
To answer questions you use a knowledge graph which contains nodes and relationships between them.
The knowledge graph is your source of truth. If you cannot find information to answer the user inquery in the knowledge graph, you must provide a response that explains the information is not available.

You have 3 tools to access the knowledge graph:
1. node_search: Use this tool to search for nodes in the knowledge graph
2. query: Use this tool to traverse the knowledge graph starting the traversal on node id(s).
3. aggregate: Use this tool to aggregate data in the knowledge graph.

You are welcome to use one or more of these tools in any order as you see fit.  However, below are some helpful tips to get the best answers.
- You can figure out what words likely correspond to node id(s) by looking at the graphSchema. 
- node_search often comes first: Users generally won't know specifies node id(s) offhand unless they provide them explicitly, but they will know conceptually what information (i.e.nodes and relationships) they are looking for. So in general, unless the user specifically provides id(s), you are usually better off first using node_search to search for the nodes they are interested in, then getting the node ids from there and running queries with them. 
- Aggregation can cause exceptions for the above.  Aggregation will sometimes be based off specific node id(s) so you should follow the same advise as above. However, some aggregations will be based off general node labels or relationship types in the graphSchema.  In this case, you should cut right to aggregate to get the data.

The following search_configs are available for node_search. when calling nodSearch please choose to best fulfill your plan to answer the users query.

search_config options:
{searchConfigs}

For query and aggregation tools, you will provide the tools with a description of the query you want to run.  Make sure to include any relevant node id(s) for that. 

## graphSchema for reference
{graphSchema}

## User Inquiry




User Inquiry:
Extract nodes, relationships and their properties where applicable from the textChunk according to the graphSchema. 
This data will be merged into a knowledge graph to support downstream search and analytics. As such:
- You must strictly adhere to the schema
- to find data to populate node ids, use the graph schema to find the node id names in the text.
- every relationship requires start and end ids, do not provide relationships with start and end ids that don't correspond to existing nodes you extracted
- You may find that not all nodes or relationship types are present in the graph schema.  That is okay. 

## TextChunk From File {fileName}
{text}

## GraphSchema
{graphSchema}
''')

