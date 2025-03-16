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

