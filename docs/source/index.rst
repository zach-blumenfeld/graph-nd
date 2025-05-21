.. Graph-ND documentation master file, created by
   sphinx-quickstart on Sun May 18 18:46:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to graph-nd documentation!
====================================
A Python package for building powerful end-to-end agentic GraphRAG systems with a simple, intuitive API.

* **Zero-to-GraphRAG in under 5 minutes** - no prior graph expertise required.
* **Built with intent to extend to production** - not just a demo tool.
* **Prioritizes support for mixed (structured + unstructured) data sources**
* **Out-of-the-Box GraphRAG Agents** - Automatically builds extensible GraphRAG agents customized to your graph schema

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   examples/index
   api/index
   contributing

Features
--------
* **Knowledge Graph Creation & Data Integration:** Easily map mixed data sources to knowledge graphs (KGs):
 - tables from CSVs, RDBMS, and Data Warehouses
 - text from PDFs and other files via KG driven entity extraction
 - other data from your own custom workflows with optimized node and relationship loading

* **Neo4j Persistence**:
 - Store KGs in Neo4j graph databases for reliability & scale
 - Leverage graph database traversals & indexing for low-latency retrieval

* **Out-of-the-Box GraphRAG Agents**
 - Automatically builds Langgraph ReAct agents with tools and system prompts customized to your graph schema and use case
 - Enables agent customization through the addition of other user defined tools and prompt context, either locally or via MCP
 - **Coming Soon** - Agent factories for creating agents in multiple other frameworks and SDKs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

