import os
from dotenv import load_dotenv
from graphrag import GraphRAG
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def main():
    # Load environment variables
    load_dotenv('.env', override=True)

    # Retrieve credentials from env variables
    uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    # Verify that required environment variables are set
    if not uri or not username or not password:
        raise ValueError("Missing required environment variables: NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD")

    # Set up Neo4j database client
    db_client = GraphDatabase.driver(uri, auth=(username, password))

    # Set up language model
    llm = ChatOpenAI(model="gpt-4o")

    # Instantiate GraphRAG
    graphrag = GraphRAG(db_client, llm)

    # Infer schema and export
    graphrag.schema.infer("a simple graph of hardware components where components "
                          "(with id, name, and description properties) can be types of or inputs to other components.")
    graphrag.schema.export('graph-schema.json')

    # Notify the user
    print("Graph schema inferred and exported to 'graph-schema.json'.")


# Run the main method when the script is executed
if __name__ == "__main__":
    main()
