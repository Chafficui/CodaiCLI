"""Tests for codaicli.tools."""

import os

import pytest

from codaicli.file_manager import FileManager
from codaicli.tools import ToolRegistry


@pytest.fixture
def project(tmp_path):
    """Create a temporary project directory with some files."""
    # Create files
    (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "app.py").write_text("import main\n\nclass App:\n    pass\n")
    (sub / "config.json").write_text('{"key": "value"}\n')
    return tmp_path


@pytest.fixture
def registry(project):
    fm = FileManager(str(project))
    return ToolRegistry(fm, str(project))


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing(self, registry, project):
        result = await registry.execute("read_file", {"path": "main.py"})
        assert not result.is_error
        assert "def hello():" in result.content
        assert "1 |" in result.content  # Line numbers

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, registry):
        result = await registry.execute("read_file", {"path": "nope.py"})
        assert result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, registry, project):
        (project / "big.py").write_text("\n".join(f"line {i}" for i in range(100)))
        result = await registry.execute(
            "read_file", {"path": "big.py", "offset": 10, "limit": 5}
        )
        assert not result.is_error
        assert "line 9" in result.content  # 0-indexed content, 1-indexed display
        assert "line 14" not in result.content  # Should stop at limit

    @pytest.mark.asyncio
    async def test_path_traversal(self, registry):
        result = await registry.execute("read_file", {"path": "../../../etc/passwd"})
        assert result.is_error
        assert "outside" in result.content.lower()


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_create_new(self, registry, project):
        result = await registry.execute(
            "write_file", {"path": "new.py", "content": "print('new')"}
        )
        assert not result.is_error
        assert "Created" in result.content
        assert (project / "new.py").read_text() == "print('new')"

    @pytest.mark.asyncio
    async def test_overwrite(self, registry, project):
        result = await registry.execute(
            "write_file", {"path": "main.py", "content": "# rewritten"}
        )
        assert not result.is_error
        assert "Overwrote" in result.content

    @pytest.mark.asyncio
    async def test_create_nested(self, registry, project):
        result = await registry.execute(
            "write_file", {"path": "deep/nested/file.py", "content": "x = 1"}
        )
        assert not result.is_error
        assert (project / "deep" / "nested" / "file.py").exists()


class TestEditFile:
    @pytest.mark.asyncio
    async def test_simple_edit(self, registry, project):
        result = await registry.execute(
            "edit_file",
            {
                "path": "main.py",
                "old_text": "print('hello')",
                "new_text": "print('world')",
            },
        )
        assert not result.is_error
        assert "world" in (project / "main.py").read_text()

    @pytest.mark.asyncio
    async def test_edit_not_found(self, registry):
        result = await registry.execute(
            "edit_file",
            {
                "path": "main.py",
                "old_text": "this does not exist",
                "new_text": "replacement",
            },
        )
        assert result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, registry):
        result = await registry.execute(
            "edit_file",
            {"path": "nope.py", "old_text": "a", "new_text": "b"},
        )
        assert result.is_error


class TestListFiles:
    @pytest.mark.asyncio
    async def test_list_root(self, registry):
        result = await registry.execute("list_files", {})
        assert not result.is_error
        assert "main.py" in result.content
        assert "utils.py" in result.content

    @pytest.mark.asyncio
    async def test_list_subdir(self, registry):
        result = await registry.execute("list_files", {"path": "src"})
        assert not result.is_error
        assert "app.py" in result.content

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, registry):
        result = await registry.execute("list_files", {"pattern": "**/*.py"})
        assert not result.is_error
        assert "main.py" in result.content
        assert "config.json" not in result.content


class TestSearchFiles:
    @pytest.mark.asyncio
    async def test_search_literal(self, registry):
        result = await registry.execute(
            "search_files", {"pattern": "def hello"}
        )
        assert not result.is_error
        assert "main.py" in result.content

    @pytest.mark.asyncio
    async def test_search_regex(self, registry):
        result = await registry.execute(
            "search_files", {"pattern": r"def \w+\("}
        )
        assert not result.is_error
        assert "main.py" in result.content
        assert "utils.py" in result.content

    @pytest.mark.asyncio
    async def test_search_with_file_pattern(self, registry):
        result = await registry.execute(
            "search_files", {"pattern": "import", "file_pattern": "*.py"}
        )
        assert not result.is_error
        assert "app.py" in result.content

    @pytest.mark.asyncio
    async def test_search_no_matches(self, registry):
        result = await registry.execute(
            "search_files", {"pattern": "zzz_nonexistent_zzz"}
        )
        assert not result.is_error
        assert "no match" in result.content.lower()


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_simple_command(self, registry):
        result = await registry.execute("run_command", {"command": "echo hello"})
        assert not result.is_error
        assert "hello" in result.content
        assert "Exit code: 0" in result.content

    @pytest.mark.asyncio
    async def test_failing_command(self, registry):
        result = await registry.execute(
            "run_command", {"command": "exit 1"}
        )
        assert not result.is_error  # Tool itself doesn't error, just reports exit code
        assert "Exit code: 1" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self, registry):
        result = await registry.execute(
            "run_command", {"command": "sleep 10", "timeout": 1}
        )
        assert "timed out" in result.content.lower()


class TestDeleteFile:
    @pytest.mark.asyncio
    async def test_delete_existing(self, registry, project):
        assert (project / "utils.py").exists()
        result = await registry.execute("delete_file", {"path": "utils.py"})
        assert not result.is_error
        assert not (project / "utils.py").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, registry):
        result = await registry.execute("delete_file", {"path": "nope.py"})
        assert result.is_error


class TestToolRegistry:
    def test_get_definitions(self, registry):
        defs = registry.get_definitions()
        names = [d.name for d in defs]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "list_files" in names
        assert "search_files" in names
        assert "run_command" in names
        assert "delete_file" in names

    def test_get_litellm_tools(self, registry):
        tools = registry.get_litellm_tools()
        assert all(t["type"] == "function" for t in tools)
        assert all("name" in t["function"] for t in tools)

    def test_is_destructive(self, registry):
        assert not registry.is_destructive("read_file")
        assert not registry.is_destructive("list_files")
        assert not registry.is_destructive("search_files")
        assert registry.is_destructive("write_file")
        assert registry.is_destructive("edit_file")
        assert registry.is_destructive("run_command")
        assert registry.is_destructive("delete_file")

    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        result = await registry.execute("nonexistent_tool", {})
        assert result.is_error
        assert "Unknown tool" in result.content
