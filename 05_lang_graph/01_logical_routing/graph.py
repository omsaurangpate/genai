from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# schema
class DetectCallResponse(BaseModel):
    is_question_ai: bool

class AiResponse(BaseModel):
    answer: str

client = OpenAI()

# State
class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool

# Node 1
def detect_query(state: State):
    user_message = state.get("user_message")

    system_prompt = """
    You are an AI assistent. Your job is to detect if the user's query is related to coding question or not.
    Return the response in specified JSON boolean only.
    """

    # OpenAI call for categorization
    llm = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=DetectCallResponse,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
    )

    state["is_coding_question"] = llm.choices[0].message.parsed.is_question_ai
    return state

# Router Edge
def route_edge(state: State) -> Literal["resolve_coding_question", "resolve_simple_query"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "resolve_coding_question"
    else:
        return "resolve_simple_query"

# Node 2
def resolve_coding_question(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are an AI assistent. Your job is to resolve user query.
    """

    # OpenAI call for categorization
    llm = client.beta.chat.completions.parse(
        model="gpt-4o",
        response_format=AiResponse,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
    )
    
    state["ai_message"] = llm.choices[0].message.parsed.answer

    return state

# Node 3
def resolve_simple_query(state: State):
    user_message = state.get("user_message")

    # OpenAI call (Simple Query)
    system_prompt = """
    You are an AI assistent. Your job is to chat with user.
    """

    # OpenAI call for categorization
    llm = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=AiResponse,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
    )
    
    state["ai_message"] = llm.choices[0].message.parsed.answer

    return state

graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("resolve_coding_question", resolve_coding_question)
graph_builder.add_node("resolve_simple_query", resolve_simple_query)
graph_builder.add_node("route_edge", route_edge)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge) # Always pass function in conditional edge
graph_builder.add_edge("resolve_coding_question", END)
graph_builder.add_edge("resolve_simple_query", END)

graph = graph_builder.compile()
