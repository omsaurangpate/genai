# Zero Shot Prompting
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

user_query = input("> ")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        { "role": "user", "content": user_query }
    ]
)

print(response.choices[0].message.content)
