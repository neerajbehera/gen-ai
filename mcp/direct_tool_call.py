#!/usr/bin/env python3
import asyncio
import json
import sys

class DirectMCPClient:
    def __init__(self, server_script):
        self.server_script = server_script
        self.proc = None
        self.request_id = 0

    async def connect(self):
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, self.server_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # Initialize session
        await self._send_request("initialize", {"protocolVersion": "0.1.0"})
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

async def main():
    if len(sys.argv) < 2:
        print("Usage: python direct_tool_call.py <tool_name> [json_arguments]")
        print("Example: python direct_tool_call.py get_weather '{\"city\":\"Tokyo\"}'")
        print("Or: python direct_tool_call.py list")
        sys.exit(1)

    client = await DirectMCPClient("weather_server.py").connect()

    if sys.argv[1] == "list":
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print(f"  {t['name']}: {t['description']}")
    else:
        tool_name = sys.argv[1]
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = await client.call_tool(tool_name, args)
        print(f"Result: {result}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())