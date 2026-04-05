# MCP Weather Server – Direct & LLM Integration

This project demonstrates the **Model Context Protocol (MCP)** with a simple weather tool.  
It includes two clients:

1. **Direct client** – calls the tool without an LLM (for testing, automation, or scripting).  
2. **LLM client** – integrates with a local Ollama model that decides when to call the tool.

Both clients communicate with the same MCP server via **JSON‑RPC over stdio**.

---


## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (only for the LLM client)
- A model pulled in Ollama (e.g., `llama3:latest`)

### 2. Setup

```bash
# Clone or create the project folder
cd mcp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install requests


# List available tools
python direct_tool_call.py list

# Call the get_weather tool
python direct_tool_call.py get_weather '{"city": "Tokyo"}'

### 1. Prerequisites
Configuration
    Edit ollama_mcp_prompt_based.py and set your preferred model:

python
    MODEL_NAME = "llama3:latest"   # Change to any model you have pulled
    Make sure Ollama is running:

bash
    ollama serve   # (usually already running)
    Run the LLM client
    bash
    python ollama_mcp_prompt_based.py "What's the weather like in Tokyo?"
    
Example output:

    text
    📋 Available MCP tools: ['get_weather']
    🟡 Asking LLM to decide tool call via prompt...
    🔧 LLM requested tool: get_weather({'city': 'Tokyo'})
    📡 MCP returned: Weather in Tokyo: 23°C, sunny

    🤖 Final answer: The weather in Tokyo is currently 23°C and sunny.