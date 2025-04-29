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
Get user input, analyse, think, again think, output with explanation, also validate the output before giving a final result.

Follow the steps in sequence:
"analyse", "think", "output", "validate", and finally "result"

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: what is 2 + 2?
Output: {{ step: "analyse", content: "Alright! the user is asking about maths query and he is asking about basic maths query." }}
Output: {{ step: "think", content: "To perform addition of two numbers I must go from left to right and add all the operands." }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "Seems like 4 is correct answer for 2 + 2." }}
Output: {{ step: "result", content: "2 + 2 = 4. The result comes from calculating the addition of 2 with 2." }}
"""

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={ "type": "json_object" },
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "Calculate 4 + 9 - 1?" },

        #
        { "role": "assistant", "content": json.dumps({ "step": "analyse", "content": "The user query is related to arithmetic operations involving addition and subtraction." }) },
        { "role": "assistant", "content": json.dumps({ "step": "think", "content": "I need to first perform the addition operation and then proceed to the subtraction to get the final result." }) },
        { "role": "assistant", "content": json.dumps({ "step": "output", "content": "The first operation is addition: 4 + 9 = 13." }) },
        { "role": "assistant", "content": json.dumps({ "step": "validate", "content": "After confirming, the addition of 4 and 9 indeed results in 13." }) },
        { "role": "assistant", "content": json.dumps({ "step": "output", "content": "Now, subtract 1 from the result of the previous operation: 13 - 1 = 12." }) },
        { "role": "assistant", "content": json.dumps({ "step": "validate", "content": "Checking the subtraction operation, 13 minus 1 correctly equals 12." }) },
        { "role": "assistant", "content": json.dumps({ "step": "result", "content": "The result of the expression 4 + 9 - 1 is 12. This result is obtained by first performing the addition (4 + 9 = 13) and then the subtraction (13 - 1 = 12)." }) },
    ]
)

print(response.choices[0].message.content)

# Final Output: The result of the expression 4 + 9 - 1 is 12. This result is obtained by first performing the addition (4 + 9 = 13) and then the subtraction (13 - 1 = 12).
