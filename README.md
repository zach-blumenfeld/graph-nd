# **Graph-ND**
*Knowledge in __Graphs, Not Documents__*

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://graph-nd.readthedocs.io/)

A Python package for building powerful end-to-end agentic [**GraphRAG**](https://neo4j.com/blog/genai/what-is-graphrag) systems with a simple, intuitive API. 

__Example Usage__: From data sources to agentic GraphRAG in 3 steps:

 ```python
# Instantiate graphrag
graphrag = GraphRAG(db_client, llm, embedding_model)

# 1) Get the graph schema. Can also define exactly via json/pydantic spec
graphrag.schema.infer("a simple graph of hardware components "
                      "where components (with id, name, and description properties)  "
                      "can be types of or inputs to other components.")

# 2) Merge data into your graph store - both structured (tables) and unstructured (docs)
graphrag.data.merge_csvs(['component-types.csv', 'component-input-output.csv'])
graphrag.data.merge_pdf('component-catalog.pdf') 

# 3) GraphRAG agent for better answers.
graphrag.agent("what sequence of components depend on silicon wafers?")

""" Output:
================================ Human Message =================================
what sequence of components depend on silicon wafers?
================================== Ai Message ==================================
Tool Calls:
  node_search (call_mhh8tLl0WbvwXxEyrS7zBxKN)
  Args:
    search_query: silicon wafers
    top_k: 5
    search_config: {'search_type': 'SEMANTIC', 'node_label': 'Component', 'search_prop': 'name'}
================================= Tool Message =================================
Name: node_search
[   {  
        "id": "N26",
        "name": "Wafer",
        "description": "Silicon wafers are the basic building block....
================================== Ai Message ==================================
Tool Calls:
  query (call_jirfr5ZDmKTL2TQRp7xx1VYJ)
  Args:
    query_instructions: Find the sequence of components that depend on the component with id 'N26' (Silicon Wafer) through 'INPUT_TO' relationships. Return the sequence of component names.
Running Query:
MATCH (start:Component {id: 'N26'})-[:INPUT_TO*]->(end:Component)
RETURN end.name AS componentName
================================= Tool Message =================================
Name: query
[   
    {"componentName": "Deposition"},
    {"componentName": "Photolithography"}, 
    ...
================================== Ai Message ==================================
The sequence of components that depend on silicon wafers is as follows:
1. Deposition
2. Photolithography
3. Etch and clean
4. Chemical mechanical planarization
5. Assembly and packaging
6. Testing
7. Finished logic chip
"""
```
## Why Graph-ND?
1. Designed to get you started with GraphRAG easily in 5 minutes. No prior graph expertise required!
2. Built with intent to extend to production - not just a demo tool. While geared for simplicity, users can customize schemas, data loading, indexes, etc.  for precision & control.
3. Prioritizes support for mixed data. Seamlessly integrates both structured (CSV, tables) and unstructured data (PDFs, text) into your knowledge graph.

## How It Works in More Detail
Here’s a step-by-step example to using graph-nd:
1. **Setup**: Instantiate and configure the GraphRAG class. GraphRAG uses Langchain under-the-hood so you can use any model(s) with Langchain support. 
``` python
from graph_nd import GraphRAG
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

db_client = GraphDatabase.driver(uri, auth=(username, password))  # Neo4j connection
embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')  # Embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)  # Language model

graphrag = GraphRAG(db_client, llm, embedding_model)
```
1. **Get the Graph Schema**: When experimenting, you can define the desired graph structure using natural language and `GraphRAG` will infer the schema automatically. When you need more precision, you can use the `schema.define` method to specify the schema exactly passing a Pydantic `GraphSchema` object. You can also `.export` & `.load` the schema to/from json files allowing you to iterate and version control the schema. 

``` python
graphrag.schema.infer("""
   A simple graph of hardware components where components 
   (with id, name, and description properties) can be types of or inputs to other components.
   """)
```
2. **Merge Data into the Graph**: Merge both structured (e.g., CSV) and unstructured (e.g., PDFs) data. The `data.merge_csvs`, `data.merge_pdf` and `data.merge_text` methods use LLMs to automatically map data to your graph following the graph schema. For cases where you need to control the mapping yourself (instead of relying on the LLM in GraphRAG), you can format your own node and relationship dict records and merge directly via the `data.merge_nodes` and `data.merge_relationships` methods. 
``` python
graphrag.data.merge_csvs(['component-types.csv', 'component-input-output.csv'])  # Structured data
graphrag.data.merge_pdf('component-catalog.pdf')  # Unstructured data
```
3. **Answer Questions with the Auto-Configured Agent**: The agent includes advanced tools for node search (full-text and semantic), graph traversals (multi-hops, paths, etc.), and aggregation queries.  These are autoconfigured based on the graph schema. For advanced use cases, `graphrag.schema.prompt_str()` serializes the graph schema with simplified query patterns. You can use this as a prompt parameter when creating your own custom chains and agent workflows.
 
``` python
# Example queries
graphrag.agent("What sequence of components depend on silicon wafers?")
graphrag.agent("Can you describe what GPUs do?")
graphrag.agent("What components have the most inputs?")
```

## Installing and Running Graph-ND

### Installation

```bash
pip install graph-nd
```

Alternatively, for development you can clone and install locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/zach-blumenfeld/graph-nd.git
   cd graph-nd
   ```

2. Install with Poetry:
   ```bash
   # Install Poetry if you haven't already
   # curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

### Configuration

1. Start a free Neo4j (Aura) instance at [console.neo4j.io/](https://console.neo4j.io/)

2. Configure your `.env` file with the following:
   ```
   NEO4J_URI=<your_neo4j_uri>
   NEO4J_USERNAME=<your_neo4j_username>
   NEO4J_PASSWORD=<your_neo4j_password>
   
   OPEN_AI_API_KEY = ... # or substitute your preferred LLM/Embedding provider(s)
   ```

## Getting Started

Explore our example notebooks to learn how to use graph-nd:

- [quickstart-example.ipynb](examples/components/quickstart-example.ipynb) - A great 101 for getting started quickly
- [retail-example](examples/retail/retail-example.ipynb) - A more advanced example showing how to add control and precision to graphrag workflows
- [companies-example](examples/companies/companies.ipynb) - GraphRAG on pre-existing graphs created externally of graph-nd


## Feedback & Contributions
We welcome your feedback and contributions to make GraphRAG better and more accessible for everyone!

If you'd like to contribute:

1. Fork the repository and create a new branch for your feature or fix
2. Squash your commits for a clean history
3. Ensure all unit tests pass (running functional and integration tests is also highly recommended)
4. Submit a pull request with a clear description of your changes

For bugs, feature requests, or questions, please open an issue on our GitHub repository.


## Running Tests
This project uses pytest for testing. Tests are organized in three categories:
- Unit tests: Basic component testing
- Integration tests: Testing interaction between components - you will need an Aura free instance and OpenAI key configured in a `.env` file.
- Functional tests: More comprehensive End-to-end testing of features - you will need an Aura free instance and OpenAI key configured in a `.env` file.

### Running Tests with Poetry

```bash
# Run all tests
poetry run pytest tests -v

# Run only unit tests
poetry run pytest tests/unit -v

# Run only integration tests
poetry run pytest tests/integration -v

# Run only functional tests
poetry run pytest tests/functional -v

# Run a specific test file
poetry run pytest tests/unit/test_specific_file.py -v

# Run a specific test function
poetry run pytest tests/unit/test_file.py::test_function -v

# Run tests with specific pattern matching
poetry run pytest tests/unit -k "pattern" -v
```

Additional pytest options:
- `-v`: Verbose output
- `-k "pattern"`: Only run tests matching the pattern
- `--tb=native`: Display full traceback
- `-x`: Stop after first failure
