"""
web_search_tool.py

A LangChain tool that lets the agent search the web via DuckDuckGo.
No API key or signup required.

Install:
    pip install ddgs
"""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (1-10). Defaults to 5.",
        ge=1,
        le=10,
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web (via DuckDuckGo) for current information, facts, "
        "news, or anything not already known. Returns a numbered list of "
        "results, each with a title, URL, and short snippet. Use this "
        "whenever the answer might depend on information beyond what's "
        "already in the conversation or the project files."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            results = DDGS().text(query, max_results=max_results)
        except RatelimitException:
            return (
                "ERROR: Rate-limited by the search backend. Wait a moment "
                "and try again, or narrow the query."
            )
        except TimeoutException:
            return "ERROR: The search request timed out. Try again."
        except DDGSException as e:
            return f"ERROR: Search failed: {e}"

        if not results:
            return f"No results found for: '{query}'."

        lines = [f"Search results for: '{query}'"]
        for i, r in enumerate(results, start=1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()
            lines.append(f"\n{i}. {title}\n   URL: {href}\n   {body}")

        return "\n".join(lines)

    async def _arun(self, query: str, max_results: int = 5) -> str:
        # ddgs is sync-only; BaseTool's default async wrapper (run in a
        # thread) is fine, so this just delegates.
        return self._run(query, max_results)