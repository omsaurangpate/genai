# Chain of Thought (CoT)
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an helpful AI assistant who is specialized in breaking down complex problems into steps and then use this steps to resolve user query.
For given user input, you analyse the input and break down the user query into multiple steps
You should atleast think 5 - 6 steps on how to resolve user query before solving it.

Follow the steps in sequence:
"analyse", "think", "output", "validate", and finally "result"

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: calculate 7 + 12 - 4?
Output: {{ step: "analyse", content: "Alright! the user is asking about maths query and he is asking about basic arithmatic operation which is addition and substraction." }}
Output: {{ step: "think", content: "To perform the given arithmatic operation I should follow Bracket, Orders, Division, Multiplication, Addition, Subtraction Rule. According to rule add 7 + 12 which is 19 and then 19 - 4 = 5" }}
Output: {{ step: "output", content: "15" }}
Output: {{ step: "validate", content: "Seems like 15 is correct answer for 7 + 12 - 4" }}
Output: {{ step: "result", content: "7 + 12 - 4 = 15. According to rule of BODMAS (Bracket, Orders, Division, Multiplication, Addition, Subtraction) rule addition comes first so we add 7 + 12 which is 19 and then substracted 19 - 4 = 15." }}
"""

messages = [
    { "role": "system", "content": system_prompt },
]

while True:
    user_query = input("> ")

    while True:
        messages.append({ "role": "user", "content": user_query })

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=messages
        )

        JSON_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(JSON_output) })

        if JSON_output.get("step") != "result":
            print(f"🧠: {JSON_output.get("content")}")
            continue

        print(f"🤖: {JSON_output.get("content")}")
        break

