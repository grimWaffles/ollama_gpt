"""
folder_tools_mcp.py

Folder/file tools for the MCP server.
The FastMCP instance is created in server.py and injected here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastmcp import FastMCP
from pydantic import Field


PROJECT_ROOT = Path(__file__).resolve().parent
OUTSIDE_ROOT = PROJECT_ROOT.parent


def register_folder_tools(mcp: FastMCP):

    def is_allowed(path: Path) -> bool:
        for root in (PROJECT_ROOT, OUTSIDE_ROOT):
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False


    def resolve_path(path: str) -> Path:
        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

        resolved = candidate.resolve()

        if not is_allowed(resolved):
            raise PermissionError(
                f"Access denied: '{resolved}' is outside allowed sandbox."
            )

        return resolved


    def read_raw(path: str) -> str:
        resolved = resolve_path(path)

        if not resolved.is_file():
            raise FileNotFoundError(
                f"'{resolved}' is not an existing file."
            )

        return resolved.read_text(encoding="utf-8")


    @mcp.tool(
        name="read_folder_or_file",
        description=(
            "Read the contents of a file or list a directory. "
            "Allowed paths include the project root, services, repo, "
            "and one level outside the project root."
        ),
    )
    def read_folder_or_file(
        path: str = Field(
            description=(
                "File or directory path. "
                "Example: 'services/llm_service.py' or '../frontend'."
            )
        )
    ) -> str:

        try:
            resolved = resolve_path(path)

        except PermissionError as e:
            return f"ERROR: {e}"


        if not resolved.exists():
            return f"ERROR: '{resolved}' does not exist."


        if resolved.is_dir():
            entries = sorted(resolved.iterdir())

            result = [
                f"Directory listing for {resolved}:"
            ]

            for entry in entries:
                suffix = "/" if entry.is_dir() else ""
                result.append(f"  {entry.name}{suffix}")

            return "\n".join(result)


        try:
            text = resolved.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            return (
                f"ERROR: '{resolved}' is not a UTF-8 text file."
            )


        numbered = "\n".join(
            f"{i + 1:>5}\t{line}"
            for i, line in enumerate(text.splitlines())
        )

        return (
            f"--- {resolved} ---\n"
            f"{numbered}"
        )


    @mcp.tool(
        name="write_folder_file",
        description=(
            "Create a new file or edit an existing file by replacing "
            "one exact text snippet. Files cannot be deleted."
        ),
    )
    def write_folder_file(
        path: str = Field(
            description="File path to create or modify."
        ),

        mode: Literal["create", "edit"] = Field(
            description=(
                "'create' writes a new file. "
                "'edit' replaces an exact existing snippet."
            )
        ),

        content: Optional[str] = Field(
            default=None,
            description=(
                "File contents when mode='create'."
            ),
        ),

        old_snippet: Optional[str] = Field(
            default=None,
            description=(
                "Exact existing text when mode='edit'."
            ),
        ),

        new_snippet: Optional[str] = Field(
            default=None,
            description=(
                "Replacement text when mode='edit'."
            ),
        ),
    ) -> str:

        try:
            resolved = resolve_path(path)

        except PermissionError as e:
            return f"ERROR: {e}"


        if mode == "create":

            if resolved.exists():
                return (
                    f"ERROR: '{resolved}' already exists."
                )

            if content is None:
                return (
                    "ERROR: content is required for create mode."
                )

            resolved.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            resolved.write_text(
                content,
                encoding="utf-8"
            )

            return (
                f"Created '{resolved}'."
            )


        if mode == "edit":

            if old_snippet is None or new_snippet is None:
                return (
                    "ERROR: old_snippet and new_snippet "
                    "are required for edit mode."
                )


            try:
                current = read_raw(str(resolved))

            except FileNotFoundError as e:
                return f"ERROR: {e}"


            count = current.count(old_snippet)

            if count == 0:
                return (
                    "ERROR: old_snippet not found. "
                    "Read the file again and copy the exact text."
                )


            if count > 1:
                return (
                    "ERROR: old_snippet matches multiple locations. "
                    "Provide more surrounding context."
                )


            updated = current.replace(
                old_snippet,
                new_snippet,
                1
            )

            resolved.write_text(
                updated,
                encoding="utf-8"
            )

            return (
                f"Edited '{resolved}'."
            )


        return (
            "ERROR: mode must be 'create' or 'edit'."
        )