#!/usr/bin/env python3
import asyncio
import json
import sys
import random

async def handle_request(request):
    method = request.get("method")
    req_id = request.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"}
                            },
                            "required": ["city"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name == "get_weather":
            city = arguments.get("city", "unknown")
            temp = random.randint(0, 35)
            condition = random.choice(["sunny", "cloudy", "rainy", "windy"])
            result_text = f"Weather in {city}: {temp}°C, {condition}"
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }

async def main():
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    
    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        response = await handle_request(request)
        # Write response as JSON line
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()   # no await, it's synchronous

if __name__ == "__main__":
    asyncio.run(main())