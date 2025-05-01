# Retrieval-Augmented Generation
# RAG Pipeline - Load (Doc / Web / ...) -> Split (Chunking) -> Vector Embeddings -> Vector DB -> Retrieve Chunks -> Similarity Search

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Environment Variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

# Load PDF
pdf_path = Path(__file__).parent / "DBMS_Notes.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Text Splitter (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)
# print(split_docs)

# Vector Embeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_key
)

## Injecting Vectors Into Qdrant DB
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     collection_name="pdf_chat",
#     url="http://localhost:6333/",
#     embedding=embedder
# )

# vector_store.add_documents(documents=split_docs)
# print("Injection Done")

# Retrieve Chunks
retriever = QdrantVectorStore.from_existing_collection(
collection_name="pdf_chat",
url="http://localhost:6333/",
embedding=embedder
)

# Client
client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

while True:
    user_query = input("> ")

    # Similarity Search
    relevent_chunks = retriever.similarity_search(
        query=user_query
    )

    system_prompt = f"""
    You are an helpful AI assistant to resolve user query. You understand the user query and gives the most accurate result on the basis of context
    You answer the user query only on the basis of context.

    context:
    {relevent_chunks}

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
