"""
folder_tools.py

Lives in the project's parent folder (the same directory that contains
./agents, ./services, ./repo, etc.).

Provides:
    - _FolderToolBase   shared sandboxing/path-resolution logic
    - FolderReadTool    read files / list directories
    - FolderWriteTool   create new files or make targeted edits to
                         existing ones (never deletes anything)

Sandbox rules (enforced by _FolderToolBase._resolve_path):
    - Anything under this file's own directory (the project root, which
      includes ./services, ./repo, ./agents, etc.) is in-bounds.
    - Anything under ONE level outside that directory (i.e. the project
      root's parent) is also in-bounds, so the agent can peek at sibling
      folders, but it can't climb any further up the tree than that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class _FolderToolBase(BaseTool):
    """Common sandbox + path-resolution logic shared by the read/write tools."""

    # This file's own directory == the project root the tools are scoped to.
    _project_root: Path = Path(__file__).resolve().parent
    # One level outside the project root.
    _outside_root: Path = _project_root.parent

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        """Resolve a user-supplied path and verify it's inside the sandbox."""
        candidate = Path(relative_or_absolute)
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        resolved = candidate.resolve()

        if not self._is_allowed(resolved):
            raise PermissionError(
                f"Access denied: '{resolved}' is outside the allowed sandbox "
                f"(must be under {self._project_root} or its parent "
                f"{self._outside_root})."
            )
        return resolved

    def _is_allowed(self, path: Path) -> bool:
        for root in (self._project_root, self._outside_root):
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False


# --------------------------------------------------------------------------- #
# READ TOOL
# --------------------------------------------------------------------------- #

class FolderReadInput(BaseModel):
    path: str = Field(
        ...,
        description=(
            "File or directory path to read. May be relative to the project "
            "root (e.g. 'services/llm_service.py', 'repo') or absolute, as "
            "long as it stays inside the allowed sandbox."
        ),
    )


class FolderReadTool(_FolderToolBase):
    name: str = "read_folder_or_file"
    description: str = (
        "Read the contents of a file (returned with line numbers) or list "
        "the contents of a directory. Use this to inspect files under "
        "./services, ./repo, or the project root. You may ALSO go one "
        "level outside the project root: pass path='..' to list what's "
        "there, then pass '../<folder_name>' (e.g. '../frontEnd') to look "
        "inside it. If a path you guess returns 'does not exist', list the "
        "parent directory first ('.' or '..') to see the real folder names "
        "before trying again."
    )
    args_schema: Type[BaseModel] = FolderReadInput

    def _run(self, path: str) -> str:
        try:
            resolved = self._resolve_path(path)
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

    # Expose a plain python helper too, so other tools (e.g. the write
    # tool) can reuse the exact same read logic without going through the
    # LangChain _run() string-formatted interface.
    def read_raw(self, path: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"'{resolved}' is not an existing file.")
        return resolved.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# WRITE TOOL
# --------------------------------------------------------------------------- #

class FolderWriteInput(BaseModel):
    path: str = Field(..., description="File path to create or edit.")
    mode: str = Field(
        ...,
        description=(
            "'create' to write a brand-new file (fails if it already "
            "exists), or 'edit' to replace one exact, unique snippet of "
            "an existing file's content with new text."
        ),
    )
    content: Optional[str] = Field(
        default=None,
        description="Full file content to write. Required when mode='create'.",
    )
    old_snippet: Optional[str] = Field(
        default=None,
        description=(
            "Exact existing text to replace. Required when mode='edit'. "
            "Must appear exactly once in the file, so include enough "
            "surrounding context to make it unique."
        ),
    )
    new_snippet: Optional[str] = Field(
        default=None,
        description="Replacement text for old_snippet. Required when mode='edit'.",
    )


class FolderWriteTool(_FolderToolBase):
    name: str = "write_folder_file"
    description: str = (
        "Create a new file, or make a targeted edit to an existing file by "
        "replacing one exact snippet of its content with new text. Has the "
        "same sandbox access as read_folder_or_file (project root, "
        "including ./services and ./repo, plus one level outside it). "
        "This tool can never delete a file."
    )
    args_schema: Type[BaseModel] = FolderWriteInput

    def __init__(self, reader: Optional[FolderReadTool] = None, **kwargs):
        super().__init__(**kwargs)
        # Reuse the read tool's logic instead of re-implementing file reads.
        object.__setattr__(self, "_reader", reader or FolderReadTool())

    def _run(
        self,
        path: str,
        mode: str,
        content: Optional[str] = None,
        old_snippet: Optional[str] = None,
        new_snippet: Optional[str] = None,
    ) -> str:
        try:
            resolved = self._resolve_path(path)
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
                current = self._reader.read_raw(str(resolved))
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