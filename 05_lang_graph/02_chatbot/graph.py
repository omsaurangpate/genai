from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
@tool()
def human_assistant_tool(query: str):
    """Request assistance from human."""
    human_response = interrupt({"query": query}) # Graph will exit out after saving data in DB
    return human_response["data"] # Resume with the data

tools = [human_assistant_tool]

llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    # messages = state.get("messages")
    # response = llm.invoke(messages)
    # return {"messages": [response]}
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": message}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot", END)

# Without any memory
graph = graph_builder.compile()

# Creates a new graph with given checkpointer
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

# graph = create_chat_graph(mongodb)
# graph = create_chat_graph(postgres)
