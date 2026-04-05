#!/usr/bin/env python3
import asyncio
import json
import sys
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3:latest"   # Use any model you have: llama2, llama3, mistral, etc.

class MCPClient:
    def __init__(self, server_script):
        self.server_script = server_script
        self.proc = None
        self.request_id = 0

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, self.server_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        resp = await self._send_request("initialize", {"protocolVersion": "0.1.0"})
        if "error" in resp:
            raise Exception(f"Init failed: {resp}")
        return self

    async def _send_request(self, method, params=None):
        self.request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        await self.proc.stdin.drain()
        line = await self.proc.stdout.readline()
        return json.loads(line)

    async def list_tools(self):
        resp = await self._send_request("tools/list")
        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name, arguments):
        resp = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        result = resp.get("result", {})
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")
        return str(result)

    async def close(self):
        if self.proc:
            self.proc.terminate()
            await self.proc.wait()

def extract_json_from_text(text):
    """Find and parse a JSON object from model output."""
    # Look for {...} pattern
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

async def chat_with_ollama(user_query, mcp_client):
    tools = await mcp_client.list_tools()
    print(f"📋 Available MCP tools: {[t['name'] for t in tools]}")
    
    # Build tool description for the prompt
    tool_descriptions = "\n".join([
        f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['inputSchema'])}"
        for t in tools
    ])
    
    system_prompt = f"""You are a helpful assistant that can use tools. You have access to these tools:

    {tool_descriptions}

    When the user asks a question that requires a tool, respond with ONLY a JSON object in the following format:
    {{"tool": "tool_name", "parameters": {{"param_name": "value"}}}}

    Do not add any extra text. Just the JSON.
    If the user's question does not require a tool, respond with a natural language answer.
    """
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    
    print("🟡 Asking LLM to decide tool call via prompt...")
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assistant_content = data["message"]["content"]
    
    # Try to parse JSON tool call
    tool_call = extract_json_from_text(assistant_content)
    if tool_call and "tool" in tool_call:
        tool_name = tool_call["tool"]
        arguments = tool_call.get("parameters", {})
        print(f"🔧 LLM requested tool: {tool_name}({arguments})")
        tool_result = await mcp_client.call_tool(tool_name, arguments)
        print(f"📡 MCP returned: {tool_result}")
        
        # Now ask the model to generate a natural answer using the tool result
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": f"The tool returned: {tool_result}\nNow answer the user's original question: {user_query}"})
        payload2 = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        }
        resp2 = requests.post(OLLAMA_URL, json=payload2)
        resp2.raise_for_status()
        final_answer = resp2.json()["message"]["content"]
        return final_answer
    else:
        # No tool call, just return the response
        return assistant_content

async def main():
    if len(sys.argv) < 2:
        print("Usage: python ollama_mcp_direct.py 'Your question here'")
        sys.exit(1)
    user_query = " ".join(sys.argv[1:])
    
    mcp = await MCPClient("weather_server.py").start()
    try:
        answer = await chat_with_ollama(user_query, mcp)
        print(f"\n🤖 Final answer: {answer}")
    finally:
        await mcp.close()

if __name__ == "__main__":
    asyncio.run(main())