# Use the Graph
from graph import graph

def call_graph():
    while True:
        user_query = input("> ")
        state = {
            "user_message": user_query,
            "ai_message": "",
            "is_coding_question": False
        }

        result = graph.invoke(state)
        print(result["ai_message"])

call_graph()
