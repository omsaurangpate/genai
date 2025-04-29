# Instruction Based Prompting
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    temperature=1,
    max_output_tokens=100,
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?"
)

print(response.output_text)
