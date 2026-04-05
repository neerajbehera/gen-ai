# test_mcp_no_sdk.py
import asyncio
import json
import sys

async def run():
    # Start the server subprocess
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "weather_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Send initialize request
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "0.1.0"}
    }
    proc.stdin.write((json.dumps(init_req) + "\n").encode())
    await proc.stdin.drain()
    
    # Read response
    line = await proc.stdout.readline()
    print("Initialize response:", line.decode())
    
    # Send tools/list request
    list_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    proc.stdin.write((json.dumps(list_req) + "\n").encode())
    await proc.stdin.drain()
    
    line = await proc.stdout.readline()
    print("Tools list response:", line.decode())
    
    proc.terminate()

asyncio.run(run())