# Hypothetical Document Embeddings
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
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
def hypothetical_doc_generator(client, user_query):
    system_prompt = """
    You are an helpful ai assistent who is specialized in generating answer on the basis of user query.

    Example: 
    Input: What is DBMS?
    Output: Database Management System (DBMS) is a software for storing and retrieving users' data while
    considering appropriate security measures. It consists of a group of programs which manipulate
    the database. The DBMS accepts the request for data from an application and instructs the
    operating system to provide the specific data. In large systems, a DBMS helps users and other
    third-party software to store and retrieve data.
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

    output = llm.choices[0].message.content
    return output

# Vector Embeddigs
def setup_embeddings(api_key):
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )
    return embedder

# Retrieve Chunks
def setup_retriever(embedder, collection_name="pdf_chat", url="http://localhost:6333/"):
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        url=url,
        embedding=embedder
    )
    return retriever

# Similarity Search
def similarity_search(retriever, output):
    relevent_chunks = retriever.similarity_search(
    query=output
    )
    return relevent_chunks

# Union of chunks
def extract_unique_chunks(relevent_chunks):
    unique_values = set()

    for chunks in relevent_chunks:
        source = chunks.metadata["source"],
        page_content = chunks.page_content

        unique_values.add(source)
        unique_values.add(page_content)
    return unique_values

# Actual Output
def generate_response(client, user_query, unique_values):
    system_prompt = f"""
    You are an helpful AI assistant to resolve user query. You understand the user query and gives the detail answer as possible on the basis of context
    You answer the user query only on the basis of context. beautify the context and then give the output result to user.

    context:
    {unique_values}

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

        # Generate Hypothetical Document
        hypothetical_generated_doc = hypothetical_doc_generator(client, user_query)

        # setup embeddings
        embedder = setup_embeddings(openai_api_key)

        # setup retriever
        retriever = setup_retriever(embedder)

        # Retrieve relevant chunks
        chunks = similarity_search(retriever, hypothetical_generated_doc)

        # Extract unique chunks
        unique_chunks = extract_unique_chunks(chunks)

        # Generate final response
        response = generate_response(client, user_query, unique_chunks)

        # Output the response
        print(response)

if __name__ == "__main__":
    main()
