"""Prompt templates for knowledge generation."""

OVERVIEW_PROMPT = """\
You are documenting a software project for a developer who needs to understand it quickly.

Given the following Python files and their docstrings/first lines, write a concise project overview.

Include:
- What the project does (1-2 sentences)
- Key entry points and commands
- Tech stack and core dependencies
- Target audience

Keep it under 300 words. Be specific, not generic.

Files:
{file_list}
"""

MODULE_PROMPT = """\
You are documenting a Python module for a developer who needs to understand it quickly.

Given the source code of `{module_path}`, write a concise module summary.

Include:
- Purpose of this module (1 sentence)
- Public classes/functions and what they do
- Key dependencies (what it imports and why)
- Important patterns or design decisions

Keep it under 250 words. Focus on what a developer needs to know to use or modify this module.

Source code:
```python
{source_code}
```
"""

ARCHITECTURE_PROMPT = """\
You are documenting the architecture of a software project.

Given the module summaries below, describe how the modules connect and interact.

Include:
- High-level data/control flow
- Module dependencies and layering
- Key integration points
- Design patterns used

Keep it under 300 words. Use concrete module names, not abstractions.

Module summaries:
{module_summaries}
"""

PATTERN_PROMPT = """\
You are identifying coding patterns and conventions in a software project.

Given the source code files below, identify recurring patterns, conventions, and idioms.

Include:
- Error handling patterns
- Configuration approach
- Testing conventions
- Code organization style
- Naming conventions

Keep it under 250 words. Be specific with examples from the code.

Source files:
{source_snippets}
"""
