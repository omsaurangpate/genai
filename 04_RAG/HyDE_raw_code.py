# Hypothetical Document Embedding
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

hypothetical_system_prompt = """
You are an helpful ai assistent who is specialized in generating answer on the basis of user query.

Example: 
Input: What is DBMS?
Output: Database Management System (DBMS) is a software for storing and retrieving users' data while
considering appropriate security measures. It consists of a group of programs which manipulate
the database. The DBMS accepts the request for data from an application and instructs the
operating system to provide the specific data. In large systems, a DBMS helps users and other
third-party software to store and retrieve data.
"""

user_query = input("> ")

llm = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={ "type": "json_object" },
        messages=[
        {
            "role": "system", 
            "content": hypothetical_system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
)

parsed_output = llm.choices[0].message.content

# Vector Embeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key
)

retriever = QdrantVectorStore.from_existing_collection(
collection_name="pdf_chat",
url="http://localhost:6333/",
embedding=embedder
)

# Similarity Search
relevent_chunks = retriever.similarity_search(
    query=parsed_output
)

unique_values = set()

for chunks in relevent_chunks:
    source = chunks.metadata["source"],
    page_content = chunks.page_content

    unique_values.add(source)
    unique_values.add(page_content)

# print(unique_values)

# Actual Query Resolution.
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

print(response.choices[0].message.content)
