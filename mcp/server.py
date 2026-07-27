from fastmcp import FastMCP

mcp = FastMCP("chat_tools")

def get_context():
    """
    Return the current request's context (vector_repo, embedding_service,
    user_id, chat_id).

    Replace this with however you actually track per-request/session state,
    e.g.:
        - FastMCP's built-in Context (from fastmcp.server.dependencies import
          get_context as fastmcp_get_context) plus ctx.get_state(...)
        - a contextvar set by your auth/transport layer
        - a lookup keyed off the incoming request/session id
    """
    raise NotImplementedError("Wire this up to your actual request context.")

