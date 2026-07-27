"""
main.py

Entry point for the MCP service. Creates the shared FastMCP app instance
and the per-request context accessor, then imports the tool modules under
./tools so their @mcp.tool-decorated functions register themselves on
this instance.

Run with:
    python main.py
"""
import asyncio
from server import mcp
import traceback
import os

print(f"DEBUG PID: {os.getpid()}")

try:
    import folder_tools_mcp as ft
    print("folder_tools_mcp: import OK")
except Exception:
    print("folder_tools_mcp: FAILED")
    traceback.print_exc()

try:
    import chat_history_search_mcp as ch
    print("chat_history_search_mcp: import OK")
except Exception:
    print("chat_history_search_mcp: FAILED")
    traceback.print_exc()

print("DEBUG registered tools:", asyncio.run(mcp.list_tools()))

if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=7000)