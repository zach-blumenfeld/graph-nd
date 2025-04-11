from enum import Enum
from typing import Dict, Any, Union


def prepare_source_metadata(
        source_metadata: Union[bool, Dict[str, Any]],
        default_metadata: Dict[str, Any],
) -> Union[Dict[str, Any], bool]:
    """
    Generates source metadata based on default values and user input.

    Args:
        default_metadata (Dict[str, Any]): Default metadata values.
        source_metadata (Union[bool, Dict[str, Any]]): User-provided metadata,
            can be True (use default), False (no metadata), or a dictionary.

    Returns:
        Dict[str, Any]: Final source metadata dictionary.
        False: If source_metadata explicitly set to `False`.

    Raises:
        ValueError: If source_metadata is of invalid type.
    """
    if source_metadata is True:
        # Use the default metadata
        return default_metadata
    elif source_metadata is False:
        # Explicitly disable metadata
        return False
    elif isinstance(source_metadata, dict):
        # Merge default metadata with provided metadata
        for k, v in default_metadata.items():
            if k not in source_metadata or not source_metadata[k]:
                source_metadata[k] = v
        return source_metadata
    else:
        # Raise error for invalid input
        raise ValueError(
            f"Invalid source_metadata value: {source_metadata}. "
            f"Must be True, False, or a dictionary."
        )


class SourceType(str, Enum):
    STRUCTURED_TABLE = "STRUCTURED_TABLE"
    STRUCTURED_TABLE_RDBMS = "STRUCTURED_TABLE_RDBMS"
    STRUCTURED_TABLE_CSV = "STRUCTURED_TABLE_CSV"
    UNSTRUCTURED_TEXT = "UNSTRUCTURED_TEXT"
    UNSTRUCTURED_TEXT_PDF_FILE = "UNSTRUCTURED_TEXT_PDF_FILE"
    NODE_LIST = "NODE_LIST"
    RELATIONSHIP_LIST = "RELATIONSHIP_LIST"
    NODE_AND_RELATIONSHIP_LISTS = "NODE_AND_RELATIONSHIP_LISTS"



class TransformType(Enum):
    LLM_TABLE_MAPPING_TO_NODE = "LLM_TABLE_MAPPING_TO_NODE"
    LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS = "LLM_TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS"
    LLM_TEXT_EXTRACTION_TO_NODES = "LLM_TEXT_EXTRACTION_TO_NODES"
    LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS = "LLM_TEXT_EXTRACTION_TO_NODES_AND_RELATIONSHIPS"
    TABLE_MAPPING_TO_NODE = "TABLE_MAPPING_TO_NODE"
    TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS = "TABLE_MAPPING_TO_NODES_AND_RELATIONSHIPS"
    UNKNOWN = "UNKNOWN"


class LoadType(Enum):
    MERGE_NODES = "MERGE_NODES"
    MERGE_RELATIONSHIPS = "MERGE_RELATIONSHIPS"
    MERGE_NODES_AND_RELATIONSHIPS = "MERGE_NODES_AND_RELATIONSHIPS"

