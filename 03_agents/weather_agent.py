# Agentic Workflow
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Tools
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    print("🛠️: Tool Called", city)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong :("

# Available Tools
available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city as an input and returns the current weather of the city."
    },
}

system_prompt = """
You are an helpful AI assistant who is specialized in resolving user query.
You work on start, plan, action, observe mode.
For the given user query and available tools, plan the step by step execution.
Based on tool selection you perform an action to call the tool.
Wait for the observation and based on the observation from the tool call resolve the user query.

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

JSON output Format:
{{
"step": "string",
"content": "string",
"function": "The name of function if the step is action",
"input": "The input parameter for the function",
}}

Available Tools:
- get_weather: Takes a city as an input and returns the current weather of the city.

Example:
User Query: What is the weather of new york?
Output: {{ "step": "plan", "content": "The user is interested in weather data of new york" }},
Output: {{ "step": "plan", "content": "From the available tool I should get_weather" }},
Output: {{ "step": "action", "function": "get_weather", "input": "new york }},
Output: {{ "step": "observe", "content": "12 Degree Cel" }},
Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degree cel." }},
"""

messages = [
    { "role": "system", "content": system_prompt },
]

user_query = input("> ")

while True:
    messages.append({ "role": "user", "content": user_query })
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=messages
    )

    parsed_output = json.loads(response.choices[0].message.content)
    messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })
    # print("PARSED OUTPUT: ", parsed_output)

    if parsed_output.get("step") == "plan":
        print(f"Planning 🧠: {parsed_output.get("content")}")
        
    if parsed_output.get("step") == "action":
        tool_name = parsed_output.get("function")
        tool_input = parsed_output.get("input")

        if available_tools.get(tool_name) != False:
            output = available_tools[tool_name].get("fn")(tool_input)
            messages.append({ "role": "assistant", "content": json.dumps(output) })
            continue

    if parsed_output.get("step") == "output":
        print(f"🤖: {parsed_output.get("content")}")
        break


# response = client.chat.completions.create(
#     model="gpt-4o",
#     response_format={ "type": "json_object" },
#     messages=[
#         { "role": "system", "content": system_prompt },
#         { "role": "user", "content": "What is the current temperature in pune?" },

#         #
#         { "role": "assistant", "content": json.dumps({ "step": "plan", "content": "The user is interested in the current temperature of Pune." }) },
#         { "role": "assistant", "content": json.dumps({ "step": "plan", "content": "From the available tools, I should use get_weather to obtain the current temperature in Pune." }) },
#         { "role": "assistant", "content": json.dumps({ "step": "action", "function": "get_weather", "input": "pune" }) },
#         { "role": "assistant", "content": json.dumps({ "step": "output", "content": "12 Degree Cel" }) },
#     ]
# )

# print(response.choices[0].message.content)
