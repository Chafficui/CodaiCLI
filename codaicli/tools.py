"""Built-in tools and tool registry for CodaiCLI agent."""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from codaicli.file_manager import FileManager
from codaicli.types import ToolDefinition, ToolResult


class ToolRegistry:
    """Manages built-in tools and their execution."""

    def __init__(self, file_manager: FileManager, project_path: str):
        self.file_manager = file_manager
        self.project_path = Path(project_path).resolve()
        self._tools: dict[str, tuple[ToolDefinition, Callable[..., str]]] = {}
        self._register_builtins()

    def register(self, definition: ToolDefinition, handler: Callable[..., str]):
        """Register a tool."""
        self._tools[definition.name] = (definition, handler)

    def get_definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions."""
        return [defn for defn, _ in self._tools.values()]

    def get_litellm_tools(self) -> list[dict]:
        """Format tools as litellm/OpenAI-compatible tool dicts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": defn.parameters,
                },
            }
            for defn in self.get_definitions()
        ]

    def is_destructive(self, name: str) -> bool:
        """Check if a tool requires user confirmation."""
        if name in self._tools:
            return self._tools[name][0].is_destructive
        return True  # Unknown tools default to destructive

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name. Runs in a thread to avoid blocking."""
        if name not in self._tools:
            return ToolResult(
                tool_call_id="",
                name=name,
                content=f"Unknown tool: {name}",
                is_error=True,
            )
        _, handler = self._tools[name]
        try:
            result = await asyncio.to_thread(handler, **arguments)
            return ToolResult(tool_call_id="", name=name, content=result)
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                name=name,
                content=f"Error: {e}",
                is_error=True,
            )

    def _validate_path(self, path: str) -> Path:
        """Resolve path and ensure it's within the project root."""
        resolved = (self.project_path / path).resolve()
        if not resolved.is_relative_to(self.project_path):
            raise ValueError(f"Path '{path}' is outside the project directory")
        return resolved

    # --- Built-in tool implementations ---

    def _register_builtins(self):
        self.register(
            ToolDefinition(
                name="read_file",
                description=(
                    "Read the contents of a file. Returns content with line numbers. "
                    "For large files, use offset and limit to read specific sections."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from the project root",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-based). Defaults to 1.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. Defaults to 500.",
                        },
                    },
                    "required": ["path"],
                },
                is_destructive=False,
            ),
            self._read_file,
        )

        self.register(
            ToolDefinition(
                name="write_file",
                description=(
                    "Create a new file or completely overwrite an existing file. "
                    "Use edit_file for targeted changes to existing files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from the project root",
                        },
                        "content": {
                            "type": "string",
                            "description": "The complete file content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
                is_destructive=True,
            ),
            self._write_file,
        )

        self.register(
            ToolDefinition(
                name="edit_file",
                description=(
                    "Make a targeted edit to an existing file using search-and-replace. "
                    "The old_text must match exactly (including whitespace and indentation). "
                    "Only the first occurrence is replaced."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from the project root",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "The exact text to find in the file",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The text to replace it with",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                is_destructive=True,
            ),
            self._edit_file,
        )

        self.register(
            ToolDefinition(
                name="list_files",
                description=(
                    "List files and directories in the project. Can list a specific "
                    "directory or match a glob pattern. Respects .codaiignore."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to project root. Defaults to '.'.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter results (e.g. '**/*.py').",
                        },
                    },
                    "required": [],
                },
                is_destructive=False,
            ),
            self._list_files,
        )

        self.register(
            ToolDefinition(
                name="search_files",
                description=(
                    "Search file contents using a regex pattern or literal string. "
                    "Returns matching lines with file paths and line numbers."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex supported)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in. Defaults to project root.",
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Glob to filter which files to search (e.g. '*.py').",
                        },
                    },
                    "required": ["pattern"],
                },
                is_destructive=False,
            ),
            self._search_files,
        )

        self.register(
            ToolDefinition(
                name="run_command",
                description=(
                    "Execute a shell command in the project directory. Returns stdout "
                    "and stderr. Use for running tests, builds, installs, etc."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Defaults to 30.",
                        },
                    },
                    "required": ["command"],
                },
                is_destructive=True,
            ),
            self._run_command,
        )

        self.register(
            ToolDefinition(
                name="delete_file",
                description="Delete a file from the project.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from the project root",
                        },
                    },
                    "required": ["path"],
                },
                is_destructive=True,
            ),
            self._delete_file,
        )

    def _read_file(self, path: str, offset: int = 1, limit: int = 500) -> str:
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not full_path.is_file():
            raise ValueError(f"Not a file: {path}")
        if self.file_manager._is_binary(full_path):
            raise ValueError(f"File appears to be binary: {path}")

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        start = max(0, offset - 1)
        end = start + limit
        lines = all_lines[start:end]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(lines, start=start + 1):
            numbered.append(f"{i:>5} | {line.rstrip()}")

        result = "\n".join(numbered)
        if end < total:
            result += f"\n\n[Truncated: showing lines {start + 1}-{end} of {total}. Use offset/limit to read more.]"

        return result

    def _write_file(self, path: str, content: str) -> str:
        full_path = self._validate_path(path)
        is_new = not full_path.exists()
        self.file_manager.create_file(path, content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        action = "Created" if is_new else "Overwrote"
        return f"{action}: {path} ({line_count} lines)"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_text)
        if count == 0:
            # Show nearby context to help the LLM fix the match
            lines = content.splitlines()
            # Try to find a partial match for better error messages
            search_snippet = old_text.splitlines()[0] if old_text else ""
            context_lines = []
            for i, line in enumerate(lines):
                if search_snippet and search_snippet.strip() in line:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines = [
                        f"{j + 1:>5} | {lines[j]}" for j in range(start, end)
                    ]
                    break

            msg = f"old_text not found in {path}."
            if context_lines:
                msg += f"\nNearest partial match:\n" + "\n".join(context_lines)
            raise ValueError(msg)

        # Replace first occurrence only
        new_content = content.replace(old_text, new_text, 1)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Generate unified diff for display
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
        diff_str = "".join(diff)
        return f"Edited {path}:\n{diff_str}" if diff_str else f"Edited {path} (no visible diff)"

    def _list_files(self, path: str = ".", pattern: str | None = None) -> str:
        target = self._validate_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not target.is_dir():
            raise ValueError(f"Not a directory: {path}")

        results: list[str] = []
        max_results = 500

        if pattern:
            for match in sorted(target.glob(pattern)):
                if len(results) >= max_results:
                    break
                if self.file_manager._is_ignored(match):
                    continue
                rel = match.relative_to(self.project_path)
                suffix = "/" if match.is_dir() else ""
                results.append(f"{rel}{suffix}")
        else:
            for root, dirs, files in os.walk(target):
                # Prune ignored directories
                dirs[:] = [
                    d
                    for d in sorted(dirs)
                    if not self.file_manager._is_ignored(Path(root) / d)
                ]
                for name in sorted(files):
                    if len(results) >= max_results:
                        break
                    fpath = Path(root) / name
                    if self.file_manager._is_ignored(fpath):
                        continue
                    results.append(str(fpath.relative_to(self.project_path)))

        total = len(results)
        output = "\n".join(results)
        if total >= max_results:
            output += f"\n\n[Showing {max_results} results. Narrow your search with a pattern.]"
        return output

    def _search_files(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str | None = None,
    ) -> str:
        target = self._validate_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        try:
            regex = re.compile(pattern)
        except re.error:
            # Fall back to literal match
            regex = re.compile(re.escape(pattern))

        matches: list[str] = []
        max_matches = 50
        files_searched = 0

        for root, dirs, files in os.walk(target):
            dirs[:] = [
                d
                for d in sorted(dirs)
                if not self.file_manager._is_ignored(Path(root) / d)
            ]
            for name in sorted(files):
                fpath = Path(root) / name
                if self.file_manager._is_ignored(fpath):
                    continue
                if file_pattern and not fnmatch.fnmatch(name, file_pattern):
                    continue
                if self.file_manager._is_binary(fpath):
                    continue

                files_searched += 1
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = fpath.relative_to(self.project_path)
                                matches.append(
                                    f"{rel}:{line_num}: {line.rstrip()}"
                                )
                                if len(matches) >= max_matches:
                                    break
                except Exception:
                    continue

                if len(matches) >= max_matches:
                    break

        output = "\n".join(matches)
        if not matches:
            output = f"No matches found for '{pattern}' in {files_searched} files."
        elif len(matches) >= max_matches:
            output += f"\n\n[Showing first {max_matches} matches. Narrow your search.]"
        return output

    def _run_command(self, command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr

            # Truncate long output
            max_chars = 10_000
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n\n[Output truncated at {max_chars} chars]"

            exit_info = f"\nExit code: {result.returncode}"
            return output + exit_info
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Error executing command: {e}"

    def _delete_file(self, path: str) -> str:
        full_path = self._validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.file_manager.delete_file(path)
        return f"Deleted: {path}"
