from typing import Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
import requests
import os
from langchain_community.utilities import GoogleSerperAPIWrapper
import gradio as gr

# Load environment variables
load_dotenv()
GOOGLE_SERPER_API_KEY = os.getenv("GOOGLE_SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# External tool clients
serper = GoogleSerperAPIWrapper()

# --- Groq LLM Helper ---
def call_groq_llm(messages, model="llama3-8b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Define the TravelState
class TravelState(TypedDict):
    messages: Annotated[List[str], operator.add]   # merge all messages across branches
    # destination: Annotated[str, pick_first]                               # single output from destination extractor
    error: Annotated[List[str], operator.add]      # merge all errors from branches
    # attractions: str                               # string from attractions node
    # weather: str                                    # string from weather node

# Node 1: Extract destination
def extract_destination_node(state: TravelState) -> TravelState:
    print("Extracting destination from messages:", state)
    user_query = state["messages"][-1]  # Get the last user message
    messages = [
        {"role": "system", "content": "Extract the travel destination from the user's message. Only return the destination name."},
        {"role": "user", "content": user_query},
    ]
    try:
        state["messages"].append(call_groq_llm(messages).strip())
    except Exception as e:
        state["error"].append(f"LLM error: {e}")
    return state

# Node 2: Search for attractions
def attractions_search_node(state: TravelState) -> TravelState:
    print("Searching for attractions in:", state['messages'][-1])
    if state.get("error"):
        return state
    try:
        result = serper.run(f"top tourist attractions in {state['messages'][-1]}")
        state["messages"].append(result)
    except Exception as e:
        state["error"].append(f"Attractions search error: {e}")
    return state

# Node 3: Search for weather
def weather_search_node(state: TravelState) -> TravelState:
    print("Searching for weather in:", state['messages'][-1])
    if state.get("error"):
        return state
    try:
        result = serper.run(f"current weather in {state['messages'][-1]}")
        state["messages"].append(result)
    except Exception as e:
        state["error"].append(f"Weather search error: {e}")
    return state

def final_response_node(state: TravelState) -> TravelState:
    print("Generating final response...")
    if state.get("error"):
        return state

    summary_prompt = [
        {"role": "system", "content": "You are a friendly travel assistant. Create a short, helpful summary for a traveler based on their question, the destination, weather, and top attractions."},
        {"role": "user", "content": f"User asked: {state['messages'][0]}"},
        {"role": "user", "content": f"Destination: {state['messages'][1]}\nAttractions: {state['messages'][4]}\nWeather: {state['messages'][5]}"},
    ]
    print("Summary prompt:", summary_prompt)
    try:
        result = (call_groq_llm(summary_prompt).strip())
        state["messages"].append(result)
    except Exception as e:
        print("Error generating final response:", e)
        state["error"].append(f"Final response generation error: {e}")
    print("Final response:", result)
    return state

# Define and compile the graph
graph_builder = StateGraph(TravelState)
graph_builder.add_node("extract_destination", extract_destination_node)
graph_builder.add_node("attractions_search", attractions_search_node)
graph_builder.add_node("weather_search", weather_search_node)
graph_builder.add_node("final_response", final_response_node)

# Add edges (with parallel branching)
graph_builder.add_edge(START, "extract_destination")
graph_builder.add_edge("extract_destination", "attractions_search")
graph_builder.add_edge("extract_destination", "weather_search")
graph_builder.add_edge("attractions_search", "final_response")
graph_builder.add_edge("weather_search", "final_response")
graph_builder.add_edge("final_response", END)


graph = graph_builder.compile()

# Function to invoke the graph
def gradio_chat(user_input):
    state = TravelState(
        messages=[user_input],
        destination="",
        attractions="",
        weather="",
        error=[],
        final_response=""
    )
    result = graph.invoke(state)

    if result["error"]:
        return "\n".join(result["error"])
    print("Final messages:", result["messages"][-1])
    return result["messages"][-1]  # Return the final response

# Sample run
gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(label="Where do you want to travel?"),
    outputs=gr.Textbox(label="Travel Recommendation"),
    title="🧳 Travel Assistant",
    description="Ask about any destination, and get current weather, top attractions, and a helpful summary."
).launch()
