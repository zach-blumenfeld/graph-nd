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

