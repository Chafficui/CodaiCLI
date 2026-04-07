"""Tests for codaicli.cli."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner

from codaicli.cli import cli, configure


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIGroup:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_no_command_runs_interactive(self, runner):
        with patch("codaicli.cli.interactive_mode") as mock:
            runner.invoke(cli)
            mock.assert_called_once()


class TestConfigure:
    @patch("codaicli.cli.Config")
    @patch("codaicli.cli.Console")
    def test_view(self, mock_console, mock_config_class, runner):
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_config.get.return_value = "test-value"
        mock_console.return_value = MagicMock()

        result = runner.invoke(configure, ["--view"])
        assert result.exit_code == 0

    @patch("codaicli.cli.Config")
    @patch("codaicli.cli.Prompt")
    def test_reset(self, mock_prompt, mock_config_class, runner):
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_prompt.ask.return_value = "y"

        result = runner.invoke(configure, ["--reset"])
        assert result.exit_code == 0
        mock_config.save.assert_called_once()


class TestInteractiveMode:
    """Tests for the async _interactive_mode function.

    These patch at the source module level since imports happen inside
    _interactive_mode(), not at cli.py module level.
    """

    @pytest.mark.asyncio
    async def test_exit_command(self):
        with (
            patch("codaicli.cli.UI") as mock_ui_class,
            patch("codaicli.cli.Config") as mock_config_class,
            patch("codaicli.file_manager.FileManager") as mock_fm,
            patch("codaicli.provider.Provider") as mock_prov,
            patch("codaicli.tools.ToolRegistry") as mock_tr,
            patch("codaicli.agent.Agent") as mock_agent_class,
        ):
            mock_ui = MagicMock()
            mock_ui_class.return_value = mock_ui
            mock_ui.get_input.return_value = "exit"
            mock_ui.console = MagicMock()

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = None

            from codaicli.cli import _interactive_mode

            await _interactive_mode()
            mock_ui.get_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_help_command(self):
        with (
            patch("codaicli.cli.UI") as mock_ui_class,
            patch("codaicli.cli.Config") as mock_config_class,
            patch("codaicli.file_manager.FileManager") as mock_fm,
            patch("codaicli.provider.Provider") as mock_prov,
            patch("codaicli.tools.ToolRegistry") as mock_tr,
            patch("codaicli.agent.Agent") as mock_agent_class,
        ):
            mock_ui = MagicMock()
            mock_ui_class.return_value = mock_ui
            mock_ui.get_input.side_effect = ["help", "exit"]
            mock_ui.console = MagicMock()

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = None

            from codaicli.cli import _interactive_mode

            await _interactive_mode()
            mock_ui.show_help.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_command(self):
        with (
            patch("codaicli.cli.UI") as mock_ui_class,
            patch("codaicli.cli.Config") as mock_config_class,
            patch("codaicli.file_manager.FileManager") as mock_fm,
            patch("codaicli.provider.Provider") as mock_prov,
            patch("codaicli.tools.ToolRegistry") as mock_tr,
            patch("codaicli.agent.Agent") as mock_agent_class,
        ):
            mock_ui = MagicMock()
            mock_ui_class.return_value = mock_ui
            mock_ui.get_input.side_effect = ["new", "exit"]
            mock_ui.console = MagicMock()

            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = None

            from codaicli.cli import _interactive_mode

            await _interactive_mode()
            mock_agent.clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_switch(self):
        with (
            patch("codaicli.cli.UI") as mock_ui_class,
            patch("codaicli.cli.Config") as mock_config_class,
            patch("codaicli.file_manager.FileManager") as mock_fm,
            patch("codaicli.provider.Provider") as mock_prov_class,
            patch("codaicli.tools.ToolRegistry") as mock_tr,
            patch("codaicli.agent.Agent") as mock_agent_class,
        ):
            mock_ui = MagicMock()
            mock_ui_class.return_value = mock_ui
            mock_ui.get_input.side_effect = ["model openai/gpt-4o", "exit"]
            mock_ui.console = MagicMock()

            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = None

            from codaicli.cli import _interactive_mode

            await _interactive_mode()
            # Provider should have been created for the new model
            assert mock_prov_class.call_count >= 2  # initial + switch

    @pytest.mark.asyncio
    async def test_query_runs_agent(self):
        with (
            patch("codaicli.cli.UI") as mock_ui_class,
            patch("codaicli.cli.Config") as mock_config_class,
            patch("codaicli.file_manager.FileManager") as mock_fm,
            patch("codaicli.provider.Provider") as mock_prov,
            patch("codaicli.tools.ToolRegistry") as mock_tr,
            patch("codaicli.agent.Agent") as mock_agent_class,
        ):
            mock_ui = MagicMock()
            mock_ui_class.return_value = mock_ui
            mock_ui.get_input.side_effect = ["what does main.py do", "exit"]
            mock_ui.console = MagicMock()

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value="It prints hello.")
            mock_agent.trim_history = MagicMock()
            mock_agent_class.return_value = mock_agent

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = None

            from codaicli.cli import _interactive_mode

            await _interactive_mode()
            mock_agent.run.assert_called_once_with("what does main.py do")
