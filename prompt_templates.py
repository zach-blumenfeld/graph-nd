from langchain_core.prompts import PromptTemplate

SCHEMA_FROM_DESC_TEMPLATE = PromptTemplate.from_template('''
Generate a graphSchema from the below description.
For any string properties that may require semantic search, include an additional textEmbedding property and
format the name as <originalPropertyName>_textembedding:

{context}
''')

SCHEMA_FROM_SAMPLE_TEMPLATE = PromptTemplate.from_template('''
Generate a graphSchema from the below sample of domain data.
For any string properties that may require semantic search, include an additional textEmbedding property and
format the name as <originalPropertyName>_textembedding:

{context}
''')

SCHEMA_FROM_DICT_TEMPLATE = PromptTemplate.from_template('''
Craft a graphSchema from the following json-like definition.  
Stay true to this definition when converting to graphSchema.  don't violate it.
For any string properties that may require semantic search, include an additional textEmbedding property and
format the name as <originalPropertyName>_textembedding:

{context}
''')

