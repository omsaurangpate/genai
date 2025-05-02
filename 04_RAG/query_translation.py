# Query Translation
# Prarallel Query (Fan Out) Retrieval

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Environment Variables
def load_environment_variables():
    load_dotenv()
    openai_api_key=os.getenv("OPENAI_API_KEY")
    gemini_api_key=os.getenv("GEMINI_API_KEY")
    return openai_api_key, gemini_api_key

# OpenAI Client
def create_openai_client(api_key, base_url):
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)

# Query Generator
def query_generator(client, user_query):
    system_prompt = """
    You are an helpful ai assistent who is specialized in generating 2 - 3 most relevent questions on the basis of user query. Always give response in strict JSON format.

    Rules:
    1. Carefully analyse the user query.
    2. Follow the strict JSON output as per Output schema.

    Output Format:
    [{{"question": string, "content": string}},{{"question": string, "content": string}},{{"question": string, "content": string}}]
    
    Example:
    Input: How to use links in HTML?
    Output: [
    {{"question": "1", "content": "What is HTML?"}},
    {{"question": "2", "content": "What is <a></a> tag in HTML?"}},
    {{"question": "3", "content": "How to use <a></a> tag in HTML"}},
    ]
    """

    llm = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    parsed_output = json.loads(llm.choices[0].message.content)
    return parsed_output

# Vector Embeddigs
def setup_embeddings(api_key):
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )
    return embedder

# Retriever
def setup_retriever(embedder, collection_name="pdf_chat", url="http://localhost:6333/"):
    # Retrieve Chunks
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        url=url,
        embedding=embedder
    )
    return retriever

# Similarity Search
def retrieve_relevant_chunks(retriever, queries):
    chunks = []
    for query in queries:
        relevent_chunks = retriever.similarity_search(
        query=query["content"]
        )
        chunks.append(relevent_chunks)
    return chunks

# Union of chunks
def extract_unique_chunks(chunks):
    unique_chunks = set()
    for chunk_group in chunks:
        for data in chunk_group:
            source = data.metadata["source"]
            page_content = data.page_content
            unique_chunks.add(source)
            unique_chunks.add(page_content)
    return unique_chunks

def generate_response(client, user_query, unique_chunks):
    system_prompt = f"""
    You are an helpful AI assistant to resolve user query. You understand the user query and gives the most accurate result on the basis of context
    You answer the user query only on the basis of context. beautify the context and then give the output result to user.

        context:
        {unique_chunks}

        Example:
        Input: What is DBMS
        Output: Database Management System (DBMS) is a software for storing and retrieving users' data while
        considering appropriate security measures. It consists of a group of programs which manipulate
        the database. The DBMS accepts the request for data from an application and instructs the
        operating system to provide the specific data. In large systems, a DBMS helps users and other
        third-party software to store and retrieve data.

        Input: What is HTML?
        Output: Your out of context. the answer of this question is not available in the pdf.
    """

    response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )

    return response.choices[0].message.content

def main():
    # load environment variables
    openai_api_key, gemini_api_key = load_environment_variables()

    # create OpenAI client (for Gemini)
    client = create_openai_client(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    while True:
        # Get user query
        user_query = input("> ")

        # Generated Related Queries
        generated_queries = query_generator(client, user_query)

        # setup embeddings
        embedder = setup_embeddings(openai_api_key)

        # setup retriever
        retriever = setup_retriever(embedder)

        # Retrieve relevant chunks
        chunks = retrieve_relevant_chunks(retriever, generated_queries)

        # Extract unique chunks
        unique_chunks = extract_unique_chunks(chunks)

        # Generate final response
        response = generate_response(client, user_query, unique_chunks)
        
        # Output the response
        print(response)

if __name__ == "__main__":
    main()