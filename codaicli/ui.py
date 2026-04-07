"""UI components for CodaiCLI."""

from __future__ import annotations

import json
import os
import re
import readline  # noqa: F401 — enables input() history (up/down arrows)
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from codaicli.types import ToolCall, ToolResult


class UI:
    """Manages UI components and output formatting."""

    def __init__(self):
        """Initialize UI components."""
        self.console = Console()
    
    def clear(self):
        """Clear the console."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_welcome(self, project_path, model: str = ""):
        """Show welcome message."""
        self.clear()
        model_line = f"\nModel: [cyan]{model}[/cyan]" if model else ""
        self.console.print(Panel.fit(
            "[bold blue]CodaiCLI[/bold blue] - [italic]AI-powered agentic CLI for code projects[/italic]\n\n"
            f"Project: [green]{project_path}[/green]"
            f"{model_line}\n\n"
            "Type your query in natural language. For example:\n"
            "- \"What does this code do?\"\n"
            "- \"Add error handling to function X\"\n"
            "- \"Run the tests and fix any failures\"\n\n"
            "Commands:\n"
            "- [bold]model <provider/model>[/bold] - Switch model (e.g. openai/gpt-4o)\n"
            "- [bold]index[/bold] - Generate/refresh project knowledge base\n"
            "- [bold]new[/bold] - Clear conversation history\n"
            "- [bold]help[/bold] - Show help\n"
            "- [bold]clear[/bold] - Clear screen\n"
            "- [bold]exit[/bold] - Exit CodaiCLI",
            title="Welcome",
            border_style="blue",
            padding=(1, 2),
        ))
    
    def show_help(self):
        """Show help information."""
        help_text = """
# CodaiCLI Help

## Commands
- `model <provider/model>` - Switch model (e.g. `model openai/gpt-4o`, `model ollama/llama3`)
- `index` - Generate/refresh project knowledge base
- `new` - Clear conversation history
- `help` - Show this help
- `clear` - Clear screen
- `exit`, `quit`, `q` - Exit CodaiCLI

## Query Examples
- "What does this code do?"
- "Find and fix the bug in the login function"
- "Add tests for the user module"
- "Run the tests and fix any failures"
- "Refactor this class to use dependency injection"
- "Create a new config file for the database"

## Features
- Agentic multi-step task execution
- 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Gemini, Ollama, Groq, Mistral, ...)
- Self-documenting knowledge base (run `index` to generate)
- MCP server integration for external tools
- Streaming responses with real-time tool execution
- File reading, writing, editing, and deletion (with confirmation)
- Shell command execution (with confirmation)
- Conversation memory within a session

## Configuration
- Config file: `~/.codaicli/config.json`
- MCP servers: `~/.codaicli/mcp_servers.json`
- API keys can be set via environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Run `codaicli configure` for interactive setup
"""

        self.console.print(Markdown(help_text))
    
    def get_input(self):
        """Get user input with readline history support (up/down arrows)."""
        self.console.print()
        try:
            return input("\033[1;34m>\033[0m: ")
        except EOFError:
            return "exit"
    
    def show_loading(self, message="Working..."):
        """Show a loading indicator."""
        return Progress()
    
    def show_response(self, response, elapsed):
        """Format and display AI response."""
        self.console.print(
            f"\n[dim]Response time: {elapsed:.2f}s[/dim]\n"
        )
        
        # Process response to highlight code blocks
        parts = re.split(r'(```[\s\S]*?```)', response)
        
        for part in parts:
            if part.startswith('```') and part.endswith('```'):
                # Code block
                language = part.split('\n')[0].replace('```', '').strip()
                code = '\n'.join(part.split('\n')[1:-1])
                
                if language == 'diff':
                    # Display diff with syntax highlighting
                    self.console.print(Syntax(code, "diff", theme="monokai"))
                else:
                    # Display other code with syntax highlighting
                    try:
                        self.console.print(Syntax(code, language or "text", theme="monokai"))
                    except Exception:
                        self.console.print(Syntax(code, "text", theme="monokai"))
            else:
                # Regular text
                if part.strip():
                    self.console.print(Markdown(part))
    
    def confirm_diff(self, file_path, diff_content):
        """Ask for confirmation to apply diff."""
        self.console.print(Panel.fit(
            f"[bold]Apply changes to:[/bold] [green]{file_path}[/green]\n\n",
            title="Confirm Changes",
            border_style="yellow"
        ))
        
        # Display the diff with syntax highlighting
        self.console.print(Syntax(diff_content, "diff", theme="monokai"))
        
        return Confirm.ask("Apply these changes?")
    
    def confirm_create(self, file_path, content):
        """Ask for confirmation to create file."""
        self.console.print(Panel.fit(
            f"[bold]Create new file:[/bold] [green]{file_path}[/green]\n\n"
            "[bold]Content:[/bold]",
            title="Confirm File Creation",
            border_style="yellow"
        ))
        
        # Try to detect language from file extension
        extension = Path(file_path).suffix.lstrip('.')
        language_map = {
            'py': 'python', 'js': 'javascript', 'ts': 'typescript',
            'html': 'html', 'css': 'css', 'json': 'json', 
            'md': 'markdown', 'txt': 'text', 'sh': 'bash'
        }
        language = language_map.get(extension, 'text')
        
        # Display the content with syntax highlighting
        self.console.print(Syntax(content, language, theme="monokai"))
        
        return Confirm.ask("Create this file?")
    
    def confirm_delete(self, file_path):
        """Ask for confirmation to delete file."""
        self.console.print(Panel.fit(
            f"[bold red]Delete file:[/bold red] [green]{file_path}[/green]",
            title="Confirm File Deletion",
            border_style="red"
        ))
        
        return Confirm.ask("Delete this file?")
    
    def confirm_run(self, command):
        """Ask for confirmation to run command."""
        self.console.print(Panel.fit(
            f"[bold]Execute command:[/bold] [green]{command}[/green]",
            title="Confirm Command Execution",
            border_style="yellow"
        ))
        
        return Confirm.ask("Execute this command?")

    # --- Agent loop UI methods ---

    def stream_text(self, text: str):
        """Print streaming text incrementally."""
        self.console.print(text, end="", highlight=False)

    def show_tool_call(self, tool_call: ToolCall):
        """Display a compact tool call notification."""
        args_summary = _summarize_args(tool_call)
        style = "red" if tool_call.name in ("run_command", "delete_file") else "yellow"
        self.console.print(
            f"  [{style}]>[/{style}] [bold]{tool_call.name}[/bold]  [dim]{args_summary}[/dim]"
        )

    def show_tool_result(self, result: ToolResult):
        """Display a brief tool result summary."""
        if result.is_error:
            # Show error in red
            truncated = result.content[:200]
            self.console.print(f"  [red]x {result.name}[/red]  [dim]{truncated}[/dim]")
        else:
            # Show brief success summary
            summary = _result_summary(result)
            self.console.print(f"  [green]✓ {result.name}[/green]  [dim]{summary}[/dim]")

    async def confirm_tool_call(self, tool_call: ToolCall) -> bool:
        """Confirm a destructive tool call. Shows details appropriate to the tool type."""
        args = tool_call.arguments

        if tool_call.name == "edit_file":
            self.console.print(Panel(
                f"[bold]Edit:[/bold] [green]{args.get('path', '?')}[/green]\n"
                f"[red]- {_truncate(args.get('old_text', ''), 200)}[/red]\n"
                f"[green]+ {_truncate(args.get('new_text', ''), 200)}[/green]",
                title="Confirm Edit",
                border_style="yellow",
            ))
        elif tool_call.name == "write_file":
            content = args.get("content", "")
            line_count = content.count("\n") + 1
            self.console.print(Panel(
                f"[bold]Write:[/bold] [green]{args.get('path', '?')}[/green] ({line_count} lines)",
                title="Confirm Write",
                border_style="yellow",
            ))
        elif tool_call.name == "run_command":
            self.console.print(Panel(
                f"[bold]Command:[/bold] [green]{args.get('command', '?')}[/green]",
                title="Confirm Command",
                border_style="red",
            ))
        elif tool_call.name == "delete_file":
            self.console.print(Panel(
                f"[bold red]Delete:[/bold red] [green]{args.get('path', '?')}[/green]",
                title="Confirm Delete",
                border_style="red",
            ))
        else:
            # Generic MCP or unknown tool
            args_display = json.dumps(args, indent=2)[:500]
            self.console.print(Panel(
                f"[bold]{tool_call.name}[/bold]\n{args_display}",
                title="Confirm Tool",
                border_style="yellow",
            ))

        return Confirm.ask("Allow?", default=True)

    def show_usage(self, usage: dict[str, int] | None):
        """Display token usage."""
        if usage:
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            self.console.print(
                f"[dim]tokens: {_format_count(input_t)} in / {_format_count(output_t)} out[/dim]"
            )


def _summarize_args(tool_call: ToolCall) -> str:
    """Create a brief summary of tool call arguments."""
    args = tool_call.arguments
    name = tool_call.name

    if name in ("read_file", "write_file", "edit_file", "delete_file"):
        return args.get("path", "?")
    if name == "list_files":
        path = args.get("path", ".")
        pattern = args.get("pattern", "")
        return f"{path} {pattern}".strip()
    if name == "search_files":
        return f"'{args.get('pattern', '?')}'"
    if name == "run_command":
        return _truncate(args.get("command", "?"), 60)

    # Generic
    return _truncate(json.dumps(args), 60)


def _result_summary(result: ToolResult) -> str:
    """Create a brief summary of a tool result."""
    content = result.content
    lines = content.count("\n")

    if result.name == "read_file":
        return f"{lines + 1} lines"
    if result.name in ("write_file", "delete_file"):
        return content.splitlines()[0] if content else ""
    if result.name == "edit_file":
        return content.splitlines()[0] if content else ""
    if result.name == "list_files":
        return f"{lines + 1} entries"
    if result.name == "search_files":
        if "No matches" in content:
            return "no matches"
        return f"{min(lines + 1, 50)} matches"
    if result.name == "run_command":
        # Show exit code if present
        for line in reversed(content.splitlines()):
            if line.startswith("Exit code:"):
                return line
        return f"{lines + 1} lines output"

    return _truncate(content, 60)


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _format_count(n: int) -> str:
    """Format a number compactly (e.g. 1234 -> 1.2k)."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"