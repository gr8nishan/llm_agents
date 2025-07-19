import gradio as gr
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
import asyncio

# It's recommended to set your API key as an environment variable
# for security and portability.
os.environ["OPENAI_API_KEY"] = "" 

# 1. Initialize the components that do not require async operations.
# This client will connect to the MongoDB MCP server.
client = MultiServerMCPClient(
    {
        "MongoDB": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "mongodb-mcp-server",
                "--connectionString",
                # Ensure this connection string is correct for your local setup.
                "url",
            ]
        }
    }
)

# Initialize the language model.
# Using gpt-4-turbo as it is a commonly available model.
model = ChatOpenAI(model="gpt-4o-mini")

# 2. Define the asynchronous function that will be called by Gradio.
async def chat_function(message, history):
    """
    This function processes the user's message using the LangGraph agent.
    
    Args:
        message (str): The user input from the Gradio interface.
        history (list): The conversation history.

    Returns:
        str: The agent's response to be displayed in the chat.
    """
    # Since client.get_tools() is async, the agent must be created inside an async function.
    tools = await client.get_tools()
    agent = create_react_agent(model=model, tools=tools)

    # Invoke the agent with the user's message.
    # The message history is automatically managed by the ReAct agent.
    response_dict = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    
    # The agent's final response is in the 'content' of the last message.
    final_response = response_dict['messages'][-1].content
    
    return final_response

# 3. Create and launch the Gradio Chat Interface.
# This provides a user-friendly web UI for interacting with the agent.
iface = gr.ChatInterface(
    fn=chat_function,
    title="LangGraph MongoDB Agent",
    description="Ask your questions about the MongoDB collections. For example: 'Tell me all mongo collections'",
    examples=[["Tell me all mongo collections"], ["List all databases"]],
    cache_examples=False,
)

if __name__ == "__main__":
    # The launch() method starts the Gradio web server.
    iface.launch()

