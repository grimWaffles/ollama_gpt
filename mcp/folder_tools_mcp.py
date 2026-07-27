"""
folder_tools_mcp.py

Tool functions for the folder-tools MCP server. The FastMCP app instance
("mcp") is created and run elsewhere (the main server module) — this file
just imports that shared instance and registers its tools on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from server import mcp

# This file's own directory == the project root the tools are scoped to.
PROJECT_ROOT = Path(__file__).resolve().parent
# One level outside the project root.
OUTSIDE_ROOT = PROJECT_ROOT.parent


def _is_allowed(path: Path) -> bool:
    for root in (PROJECT_ROOT, OUTSIDE_ROOT):
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_path(relative_or_absolute: str) -> Path:
    """Resolve a user-supplied path and verify it's inside the sandbox."""
    candidate = Path(relative_or_absolute)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()

    if not _is_allowed(resolved):
        raise PermissionError(
            f"Access denied: '{resolved}' is outside the allowed sandbox "
            f"(must be under {PROJECT_ROOT} or its parent {OUTSIDE_ROOT})."
        )
    return resolved


def _read_raw(path: str) -> str:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"'{resolved}' is not an existing file.")
    return resolved.read_text(encoding="utf-8")


@mcp.tool(
    name="read_folder_or_file",
    description=(
        "Read the contents of a file (returned with line numbers) or list "
        "the contents of a directory. Use this to inspect files under "
        "./services, ./repo, or the project root. You may ALSO go one "
        "level outside the project root: pass path='..' to list what's "
        "there, then pass '../<folder_name>' (e.g. '../frontEnd') to look "
        "inside it. If a path you guess returns 'does not exist', list the "
        "parent directory first ('.' or '..') to see the real folder names "
        "before trying again."
    ),
)
def read_folder_or_file(
    path: str = Field(
        description=(
            "File or directory path to read. May be relative to the "
            "project root (e.g. 'services/llm_service.py', 'repo') or "
            "absolute, as long as it stays inside the allowed sandbox."
        )
    )
) -> str:
    try:
        resolved = _resolve_path(path)
    except PermissionError as e:
        return f"ERROR: {e}"

    if not resolved.exists():
        hint_parent = resolved.parent
        hint = ""
        if hint_parent.exists() and hint_parent.is_dir():
            try:
                names = ", ".join(sorted(p.name for p in hint_parent.iterdir()))
                hint = f" Try listing '{hint_parent}' first — it contains: {names}."
            except PermissionError:
                pass
        return f"ERROR: '{resolved}' does not exist.{hint}"

    if resolved.is_dir():
        entries = sorted(resolved.iterdir())
        lines = [f"Directory listing for {resolved}:"]
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"  {entry.name}{suffix}")
        return "\n".join(lines)

    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: '{resolved}' is not a UTF-8 text file."

    numbered = "\n".join(
        f"{i + 1:>5}\t{line}" for i, line in enumerate(text.splitlines())
    )
    return f"--- {resolved} ---\n{numbered}"


@mcp.tool(
    name="write_folder_file",
    description=(
        "Create a new file, or make a targeted edit to an existing file by "
        "replacing one exact snippet of its content with new text. Has the "
        "same sandbox access as read_folder_or_file (project root, "
        "including ./services and ./repo, plus one level outside it). "
        "This tool can never delete a file."
    ),
)
def write_folder_file(
    path: str = Field(description="File path to create or edit."),
    mode: Literal["create", "edit"] = Field(
        description=(
            "'create' to write a brand-new file (fails if it already "
            "exists), or 'edit' to replace one exact, unique snippet of "
            "an existing file's content with new text."
        )
    ),
    content: Optional[str] = Field(
        default=None,
        description="Full file content to write. Required when mode='create'.",
    ),
    old_snippet: Optional[str] = Field(
        default=None,
        description=(
            "Exact existing text to replace. Required when mode='edit'. "
            "Must appear exactly once in the file, so include enough "
            "surrounding context to make it unique."
        ),
    ),
    new_snippet: Optional[str] = Field(
        default=None,
        description="Replacement text for old_snippet. Required when mode='edit'.",
    ),
) -> str:
    try:
        resolved = _resolve_path(path)
    except PermissionError as e:
        return f"ERROR: {e}"

    if mode == "create":
        if resolved.exists():
            return (
                f"ERROR: '{resolved}' already exists. Use mode='edit' "
                f"to modify it instead."
            )
        if content is None:
            return "ERROR: 'content' is required when mode='create'."
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Created '{resolved}' ({len(content)} chars)."

    if mode == "edit":
        if old_snippet is None or new_snippet is None:
            return "ERROR: 'old_snippet' and 'new_snippet' are required when mode='edit'."
        try:
            current = _read_raw(str(resolved))
        except FileNotFoundError as e:
            return f"ERROR: {e}"

        occurrences = current.count(old_snippet)
        if occurrences == 0:
            return (
                "ERROR: old_snippet not found in the file. Re-read the "
                "file with read_folder_or_file and copy the snippet "
                "exactly."
            )
        if occurrences > 1:
            return (
                f"ERROR: old_snippet matches {occurrences} places in the "
                f"file. Include more surrounding context so it matches "
                f"exactly once."
            )

        updated = current.replace(old_snippet, new_snippet, 1)
        resolved.write_text(updated, encoding="utf-8")
        return f"Edited '{resolved}': replaced 1 occurrence."

    return "ERROR: 'mode' must be either 'create' or 'edit'."