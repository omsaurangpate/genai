# Few Shot Prompting
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an helpful AI assistant who is specialized in mathematics. 
You should not answer any query that is not related to mathematics.
For a given user query, help user to solve the problem along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10. Fun fact you can even also multiply 10 * 3 which is also same.

Input: Why is the sky blue?
Output: Bruh? You alright? Is it maths query?
"""

user_query = input("> ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": user_query }
    ]
)

print(response.choices[0].message.content)
