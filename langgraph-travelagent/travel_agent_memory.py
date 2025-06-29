from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import gradio as gr
import os
import uuid

# --- Load .env and LLM ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
llm = init_chat_model("groq:llama3-8b-8192")

# --- MemorySaver ---
saver = MemorySaver()

# --- Shared State ---
class TravelState(TypedDict):
    messages: Annotated[List, add_messages]

# --- Tools ---
@tool
def search_attractions(destination: str) -> str:
    """Find tourist attractions in a location."""
    return GoogleSerperAPIWrapper().run(f"top tourist attractions in {destination}")

@tool
def search_weather(destination: str) -> str:
    """Get current weather in a location."""
    return GoogleSerperAPIWrapper().run(f"current weather in {destination}")

@tool
def search_flights(route: str) -> str:
    """Find flights like 'flights from Delhi to Paris'."""
    return GoogleSerperAPIWrapper().run(f"{route} flights")

@tool
def search_restaurants(location: str) -> str:
    """Find good restaurants in a location."""
    return GoogleSerperAPIWrapper().run(f"best restaurants in {location}")


tools = [search_attractions, search_weather, search_flights, search_restaurants]
llm_with_tools = llm.bind_tools(tools)

# --- System Prompt ---
SYSTEM_PROMPT = HumanMessage(content=(
    "You are a helpful travel assistant. Use tools when appropriate:\n"
    "- Use 'search_weather' for weather\n"
    "- Use 'search_attractions' for sightseeing\n"
    "- Use 'search_flights' for flights\n"
    "- Use 'search_restaurants' for food\n"
    "After using each tool, explicitly include the tool result in your reply."
))

# --- Chatbot Node ---
def chatbot_node(state: TravelState) -> TravelState:
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    assert len(messages.tool_calls) <= 1
    return {"messages": state["messages"] + [response]}

# --- Build the Graph ---
graph_builder = StateGraph(TravelState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)


graph = graph_builder.compile(checkpointer=saver)



# --- Execution Function ---
def run_graph(user_input: str, session_id: str) -> str:
    state = {"messages": [HumanMessage(content=user_input)]}
    result = graph.invoke(state, config={"configurable": {"thread_id": session_id, "checkpoint": saver}})
    return result["messages"][-1].content

# --- Gradio Interface ---
session_id = str(uuid.uuid4())[:8]

chat_history = []

def gradio_chat(user_input, chat_history_state):
    # Add user message to state
    chat_history_state.append((user_input, None))

    # Run the graph
    result = run_graph(user_input, session_id)

    # Add assistant response to state
    chat_history_state[-1] = (user_input, result)
    return chat_history_state, chat_history_state

def reset_chat():
    saver.delete_state(session_id)
    return [], []

with gr.Blocks() as demo:
    gr.Markdown("🧳 **Travel Assistant with Memory**")

    chatbot = gr.Chatbot(label="TravelBot", height=400)
    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(placeholder="Ask about travel, flights, weather...", scale=4)
        reset_btn = gr.Button("Reset", scale=1)

    user_input.submit(fn=gradio_chat, inputs=[user_input, state], outputs=[chatbot, state])
    reset_btn.click(fn=reset_chat, outputs=[chatbot, state])

demo.launch()