import os
from dotenv import load_dotenv
from graph_nd import GraphRAG
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

    # Set up embeddings and language model
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    llm = ChatOpenAI(model="gpt-4o", temperature = 0.0)

    # Instantiate GraphRAG
    graphrag = GraphRAG(db_client, llm, embeddings)

    # get graph schema
    graphrag.schema.load('graph-schema.json')

    # ask a question
    question = "what sequence of components depend on silicon wafers?"
    print(f"Question: {question}")
    graphrag.agent(question)


# Run the main method when the script is executed
if __name__ == "__main__":
    main()
