# Vector Embeddings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

input_text = "The cat sat on the mat."

response = client.embeddings.create(
    input=input_text,
    model="text-embedding-3-small"
)

print("Input Text:", input_text, "| Vector Embeddings: ", response.data[0].embedding)
