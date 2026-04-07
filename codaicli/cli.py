#!/usr/bin/env python3
"""CodaiCLI - AI-powered agentic CLI assistant for code projects."""

from __future__ import annotations

import asyncio
import json
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from codaicli.config import Config
from codaicli.ui import UI


@click.group(invoke_without_command=True)
@click.version_option()
@click.pass_context
def cli(ctx):
    """CodaiCLI - AI-powered agentic CLI assistant for code projects."""
    if ctx.invoked_subcommand is None:
        interactive_mode()


@cli.command()
@click.option("--view", is_flag=True, help="View current configuration")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults")
def configure(view, reset):
    """Configure model, API keys, and settings."""
    console = Console()
    config = Config()

    if view:
        _show_configuration(console, config)
        return

    if reset:
        if Prompt.ask("Reset all configuration?", choices=["y", "n"]) == "y":
            config.config = {}
            config.save()
            console.print("[green]Configuration reset.[/green]")
        return

    console.print(
        Panel.fit(
            "[bold blue]CodaiCLI Configuration[/bold blue]\n\n"
            "Configure your model and API settings."
        )
    )

    sections = [
        ("Model", _configure_model, "Set the LLM model (e.g. openai/gpt-4o)"),
        ("API Key", _configure_api_key, "Set API key for the active provider"),
        ("API Base URL", _configure_api_base, "Set custom API base (for Ollama, etc.)"),
        ("Advanced", _configure_advanced, "Max tokens, temperature"),
    ]

    while True:
        _show_configuration(console, config)

        console.print("\n[bold cyan]Configuration Menu[/bold cyan]")
        for i, (name, _, desc) in enumerate(sections, 1):
            console.print(f"{i}. {name} - {desc}")
        console.print(f"{len(sections) + 1}. Save and Exit")
        console.print(f"{len(sections) + 2}. Exit without saving")

        choices = [str(i) for i in range(1, len(sections) + 3)]
        choice = Prompt.ask("\nSelect an option", choices=choices, default=str(len(sections) + 1))

        if choice == str(len(sections) + 1):
            config.save()
            console.print("\n[green]Configuration saved![/green]")
            break
        elif choice == str(len(sections) + 2):
            if Prompt.ask("Exit without saving?", choices=["y", "n"]) == "y":
                break
        else:
            name, func, _ = sections[int(choice) - 1]
            console.print(f"\n[bold cyan]{name}[/bold cyan]")
            func(console, config)


def _show_configuration(console: Console, config: Config):
    """Display current configuration."""
    console.print(Panel.fit("[bold blue]Current Configuration[/bold blue]"))
    console.print(f"\n[bold cyan]Model:[/bold cyan] {config.get('model', 'Not set')}")

    api_key = config.get("api_key", "")
    console.print(f"[bold cyan]API Key:[/bold cyan] {'••••••••' if api_key else 'Not set (using env vars)'}")

    api_base = config.get("api_base", "")
    if api_base:
        console.print(f"[bold cyan]API Base:[/bold cyan] {api_base}")

    console.print(f"[bold cyan]Max Tokens:[/bold cyan] {config.get('max_tokens', 4096)}")
    console.print(f"[bold cyan]Temperature:[/bold cyan] {config.get('temperature', 0.2)}")


def _configure_model(console: Console, config: Config):
    """Configure the LLM model."""
    console.print(
        "\nUse litellm format: [bold]provider/model[/bold]"
        "\nExamples: openai/gpt-4o, anthropic/claude-sonnet-4-20250514, "
        "ollama/llama3, groq/mixtral-8x7b-32768"
    )
    model = Prompt.ask(
        "Model",
        default=config.get("model", "anthropic/claude-sonnet-4-20250514"),
    )
    config.set("model", model)


def _configure_api_key(console: Console, config: Config):
    """Configure the API key."""
    console.print(
        "\nSet the API key for your provider. "
        "You can also use environment variables:\n"
        "  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc."
    )
    api_key = Prompt.ask("API key (leave empty to use env vars)", password=True, default="")
    config.set("api_key", api_key if api_key else None)


def _configure_api_base(console: Console, config: Config):
    """Configure custom API base URL."""
    console.print(
        "\nSet a custom API base URL (for Ollama, local models, etc.).\n"
        "  Ollama default: http://localhost:11434"
    )
    api_base = Prompt.ask(
        "API base URL (leave empty for default)",
        default=config.get("api_base", ""),
    )
    config.set("api_base", api_base if api_base else None)


def _configure_advanced(console: Console, config: Config):
    """Configure advanced settings."""
    max_tokens = Prompt.ask(
        "Max tokens per response",
        default=str(config.get("max_tokens", 4096)),
    )
    try:
        config.set("max_tokens", int(max_tokens))
    except ValueError:
        console.print("[yellow]Invalid value, keeping current.[/yellow]")

    temperature = Prompt.ask(
        "Temperature (0.0 to 1.0)",
        default=str(config.get("temperature", 0.2)),
    )
    try:
        temp = float(temperature)
        if 0 <= temp <= 1:
            config.set("temperature", temp)
        else:
            console.print("[yellow]Must be 0-1, keeping current.[/yellow]")
    except ValueError:
        console.print("[yellow]Invalid value, keeping current.[/yellow]")


async def _interactive_mode():
    """Async main interactive loop."""
    ui = UI()
    config = Config()

    project_path = os.getcwd()

    # Initialize provider
    model = config.get("model", "anthropic/claude-sonnet-4-20250514")

    from codaicli.file_manager import FileManager
    from codaicli.provider import Provider
    from codaicli.tools import ToolRegistry

    file_manager = FileManager(project_path)
    tool_registry = ToolRegistry(file_manager, project_path)

    provider = Provider(
        model=model,
        api_key=config.get("api_key"),
        api_base=config.get("api_base"),
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("max_tokens", 4096),
    )

    # MCP (optional)
    mcp_manager = None
    try:
        from codaicli.mcp_client import MCPManager

        mcp_manager = MCPManager()
        mcp_config = mcp_manager.load_config()
        for name, server_config in mcp_config.get("mcpServers", {}).items():
            try:
                await mcp_manager.connect(name, server_config)
                ui.console.print(f"  [dim]MCP: connected to {name}[/dim]")
            except Exception as e:
                ui.console.print(f"  [dim yellow]MCP: failed to connect to {name}: {e}[/dim yellow]")
    except ImportError:
        pass  # MCP not installed, that's fine

    from codaicli.agent import Agent

    agent = Agent(
        provider=provider,
        tool_registry=tool_registry,
        mcp_manager=mcp_manager,
        on_text=ui.stream_text,
        on_tool_call=ui.show_tool_call,
        on_tool_result=ui.show_tool_result,
        confirm_tool=ui.confirm_tool_call,
    )

    ui.show_welcome(project_path, model=model)

    try:
        while True:
            query = ui.get_input()

            if not query.strip():
                continue

            lower = query.lower().strip()

            if lower in ("exit", "quit", "q"):
                ui.console.print("[bold blue]Goodbye![/bold blue]")
                break

            if lower in ("clear", "cls"):
                ui.clear()
                continue

            if lower == "help":
                ui.show_help()
                continue

            if lower == "new":
                agent.clear_history()
                ui.console.print("[green]Conversation cleared.[/green]")
                continue

            if lower.startswith("model "):
                new_model = query[6:].strip()
                if new_model:
                    agent.provider = Provider(
                        model=new_model,
                        api_key=config.get("api_key"),
                        api_base=config.get("api_base"),
                        temperature=config.get("temperature", 0.2),
                        max_tokens=config.get("max_tokens", 4096),
                    )
                    ui.console.print(f"[green]Switched to {new_model}[/green]")
                else:
                    ui.console.print(f"[dim]Current model: {agent.provider.model}[/dim]")
                continue

            # Run the agent
            try:
                agent.trim_history()
                await agent.run(query)
                ui.console.print()  # newline after streaming
            except KeyboardInterrupt:
                ui.console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as e:
                ui.console.print(f"[bold red]Error:[/bold red] {e}")

    finally:
        # Clean shutdown
        if mcp_manager:
            await mcp_manager.disconnect_all()


def interactive_mode():
    """Entry point for interactive mode (sync wrapper)."""
    asyncio.run(_interactive_mode())


if __name__ == "__main__":
    cli()
