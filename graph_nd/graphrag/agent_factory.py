from typing import Sequence, Callable, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from graph_nd.graphrag import GraphRAG
from graph_nd.graphrag.prompt_templates import AGENT_SYSTEM_TEMPLATE
from google.adk.agents import Agent

def create_adk_agent(graphrag: GraphRAG, model:str, name:str, instruction:Optional[str]=None, tools=None):
    if tools:
        tools = [graphrag.node_search, graphrag.query, graphrag.aggregate] + tools
    else:
        tools = [graphrag.node_search, graphrag.query, graphrag.aggregate]
    if instruction:
        additional_instruction = "## Additional Instructions\n" + instruction
    else:
        additional_instruction = ""

    inst = AGENT_SYSTEM_TEMPLATE.invoke({'searchConfigs': graphrag.get_search_configs_prompt(),
                                                     'graphSchema': graphrag.schema.schema.prompt_str(),
                                                     'additionalInstructions': additional_instruction}).to_string()
    return Agent(model=model, name=name, instruction=inst, tools=tools)

def create_langgraph_agent(graphrag: GraphRAG, **kwargs):
    """
    A factory for creating Langgraph Agents backed with GraphRAG and Knowledge Graph

    Arguments:
        **kwargs: Keyword arguments passed to the original `create_react_agent`.

    Returns:
        The result of invoking `create_react_agent`.
    """
    if "tools" in kwargs:
        other_tools = kwargs["tools"]
        if not isinstance(other_tools, Sequence):
            raise ValueError(f"'tools' must be a Sequence, but got {type(other_tools).__name__} instead.")
        for index, item in enumerate(other_tools):
            if not isinstance(item, (BaseTool, Callable)):
                raise ValueError(
                    f"Invalid item at index {index}: {item} (type: {type(item).__name__}). "
                    f"'tools' must only contain instances of BaseTool or Callable."
                )
    else:
        other_tools = []

    # Combine tools
    all_tools = [graphrag.node_search, graphrag.query, graphrag.aggregate] + other_tools

    # Inject the combined tools into `kwargs`
    kwargs["tools"] = all_tools

    # Inject prompt into kwargs
    if "prompt" in kwargs:
        if kwargs["prompt"] is None or not isinstance(kwargs["prompt"], str):
            raise ValueError("`prompt` must be a non-null string.")
        additional_instruction = "## Additional Instructions\n" + kwargs["prompt"]
    else:
        additional_instruction = ""  # Default to an empty string if `prompt` key is not in kwargs

    # Get model if not included
    if "model" not in kwargs:
        kwargs["model"] = graphrag.llm

    kwargs["prompt"] = AGENT_SYSTEM_TEMPLATE.invoke({'searchConfigs': graphrag.get_search_configs_prompt(),
                                                     'graphSchema': graphrag.schema.schema.prompt_str(),
                                                     'additionalInstructions': additional_instruction}).to_string()

    return create_react_agent(**kwargs)