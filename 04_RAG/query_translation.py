# Query Translation
# Prarallel Query (Fan Out) Retrieval

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

openai_api_key=os.getenv("OPENAI_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def query_generator(user_query: str):
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

user_query = input("> ")

generated_queries = query_generator(user_query)

chunks = []
for querie in generated_queries:
    # Vector Embeddings
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=openai_api_key
    )

    # Retrieve Chunks
    retriever = QdrantVectorStore.from_existing_collection(
    collection_name="pdf_chat",
    url="http://localhost:6333/",
    embedding=embedder
    )

    # Similarity Search
    relevent_chunks = retriever.similarity_search(
        query=querie["content"]
    )

    chunks.append(relevent_chunks)

unique_chunks = set()

for chunk in chunks:
    for data in chunk:
        source = data.metadata["source"]
        page_content = data.page_content
    
    unique_chunks.add(source)
    unique_chunks.add(page_content)


system_prompt = """
You are an helpful AI assistant to resolve user query. You understand the user query and gives the most accurate result on the basis of context
    You answer the user query only on the basis of context.

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

print(response.choices[0].message.content)