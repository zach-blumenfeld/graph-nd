from langchain_core.prompts import PromptTemplate


NODE_MAPPING_FROM_DIR_TEMPLATE = PromptTemplate.from_template('''
The tableSchema represents the schema for a table entity that we need to map to a node in the below graphSchema. 

## Additional Directions
{directions}

## tableSchema
Table Name: {tableName}
{tableSchema}

## graphSchema
{graphSchema}
''')

RELATIONSHIPS_MAPPING_FROM_DIR_TEMPLATE = PromptTemplate.from_template('''
The tableSchema represents the schema for a table for 1 or more relationships between entities that we need to map to the below graphSchema.

## Additional Directions
{directions}

## tableSchema
Table Name: {tableName}
{tableSchema}

## graphSchema
{graphSchema}
''')

SCHEMA_MAPPING_DIRECTIVES_TEMPLATE = PromptTemplate.from_template('''
Generate mapping directions from the below descriptionOfUseCase, sourceDataModels, and target graphSchema.
YOu do not need to map everything in the source data models. Take a minimalist approach for mapping to accomplish use cases.  Less is better. 

## descriptionOfUseCase:
{useCase}

## sourceDataModels:
{sourceDataModels}

## graphSchema
{graphSchema}
''')

