import requests

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

output = available_tools["get_weather"].get("fn")("Pune")
print(output)
