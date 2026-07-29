from langchain_mcp_adapters.client import MultiServerMCPClient


class McpToolService:
    def __init__(self, server_url):
        self.mcp_tool_client = MultiServerMCPClient(
            {
                "chat_tools": {
                    "transport": "streamable_http",
                    "url": server_url,
                }
            }
        )
