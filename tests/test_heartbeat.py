"""Tests for deepclaw.heartbeat module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepclaw.config import HeartbeatConfig
from deepclaw.heartbeat import (
    HEARTBEAT_OK,
    HeartbeatRunner,
    is_checklist_empty,
    is_heartbeat_ok,
    is_quiet_hours,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> HeartbeatConfig:
    defaults = {
        "enabled": True,
        "interval_minutes": 30,
        "quiet_hours_start": None,
        "quiet_hours_end": None,
        "timezone": "UTC",
        "max_failures": 3,
        "notify_chat_id": "123",
    }
    defaults.update(overrides)
    return HeartbeatConfig(**defaults)


# ---------------------------------------------------------------------------
# is_checklist_empty
# ---------------------------------------------------------------------------


class TestIsChecklistEmpty:
    def test_empty_string(self):
        assert is_checklist_empty("") is True

    def test_whitespace_only(self):
        assert is_checklist_empty("   \n\n  ") is True

    def test_headers_only(self):
        assert is_checklist_empty("# Heartbeat Checklist\n## Section\n") is True

    def test_html_comments_only(self):
        content = "<!-- This is a comment -->\n<!-- Another comment -->"
        assert is_checklist_empty(content) is True

    def test_empty_checkboxes_only(self):
        content = "- [ ]\n- [x]\n* [ ]\n"
        assert is_checklist_empty(content) is True

    def test_bare_list_markers_only(self):
        content = "-\n-\n*\n"
        assert is_checklist_empty(content) is True

    def test_template_seed(self):
        from deepclaw.heartbeat import DEFAULT_HEARTBEAT_SEED
        assert is_checklist_empty(DEFAULT_HEARTBEAT_SEED) is True

    def test_headers_and_comments_mixed(self):
        content = "# Heartbeat\n<!-- comment -->\n## Checks\n"
        assert is_checklist_empty(content) is True

    def test_actual_content(self):
        content = "# Checklist\n- Check disk usage above 90%\n"
        assert is_checklist_empty(content) is False

    def test_checkbox_with_text(self):
        content = "- [ ] Check if nginx is running\n"
        assert is_checklist_empty(content) is False

    def test_plain_text(self):
        assert is_checklist_empty("Check the server logs") is False


# ---------------------------------------------------------------------------
# is_heartbeat_ok
# ---------------------------------------------------------------------------


class TestIsHeartbeatOk:
    def test_exact_match(self):
        assert is_heartbeat_ok(HEARTBEAT_OK) is True

    def test_with_surrounding_text(self):
        assert is_heartbeat_ok(f"Everything looks fine. {HEARTBEAT_OK}") is True

    def test_with_newlines(self):
        assert is_heartbeat_ok(f"Checked all items.\n\n{HEARTBEAT_OK}\n") is True

    def test_no_match(self):
        assert is_heartbeat_ok("Everything is fine, nothing to report.") is False

    def test_empty_response(self):
        assert is_heartbeat_ok("") is False

    def test_partial_match(self):
        assert is_heartbeat_ok("HEARTBEAT_O") is False


# ---------------------------------------------------------------------------
# is_quiet_hours
# ---------------------------------------------------------------------------


class TestIsQuietHours:
    def test_no_quiet_hours_configured(self):
        cfg = _config(quiet_hours_start=None, quiet_hours_end=None)
        assert is_quiet_hours(cfg) is False

    def test_only_start_configured(self):
        cfg = _config(quiet_hours_start=23, quiet_hours_end=None)
        assert is_quiet_hours(cfg) is False

    def test_only_end_configured(self):
        cfg = _config(quiet_hours_start=None, quiet_hours_end=8)
        assert is_quiet_hours(cfg) is False

    @patch("deepclaw.heartbeat.datetime")
    def test_simple_range_inside(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 3, 0, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=1, quiet_hours_end=6)
        assert is_quiet_hours(cfg) is True

    @patch("deepclaw.heartbeat.datetime")
    def test_simple_range_outside(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 10, 0, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=1, quiet_hours_end=6)
        assert is_quiet_hours(cfg) is False

    @patch("deepclaw.heartbeat.datetime")
    def test_midnight_wrap_inside_late(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 23, 30, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=23, quiet_hours_end=8)
        assert is_quiet_hours(cfg) is True

    @patch("deepclaw.heartbeat.datetime")
    def test_midnight_wrap_inside_early(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 5, 0, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=23, quiet_hours_end=8)
        assert is_quiet_hours(cfg) is True

    @patch("deepclaw.heartbeat.datetime")
    def test_midnight_wrap_outside(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=23, quiet_hours_end=8)
        assert is_quiet_hours(cfg) is False

    @patch("deepclaw.heartbeat.datetime")
    def test_invalid_timezone_defaults_to_utc(self, mock_dt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        mock_dt.now.return_value = datetime(2026, 3, 28, 3, 0, tzinfo=ZoneInfo("UTC"))
        cfg = _config(quiet_hours_start=1, quiet_hours_end=6, timezone="Invalid/Zone")
        assert is_quiet_hours(cfg) is True


# ---------------------------------------------------------------------------
# HeartbeatRunner
# ---------------------------------------------------------------------------


class TestHeartbeatRunner:
    @pytest.fixture
    def runner(self):
        cfg = _config(enabled=True, notify_chat_id="12345")
        agent = AsyncMock()
        channel = AsyncMock()
        return HeartbeatRunner(cfg, agent, {"telegram": channel})

    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        cfg = _config(enabled=False)
        runner = HeartbeatRunner(cfg, AsyncMock(), {})
        await runner.start()
        assert runner._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, runner):
        await runner.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_tick_skips_during_quiet_hours(self, runner):
        with patch("deepclaw.heartbeat.is_quiet_hours", return_value=True):
            await runner._tick()
        runner._agent.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_skips_empty_checklist(self, runner):
        with patch("deepclaw.heartbeat._load_checklist", return_value="<!-- empty -->"):
            await runner._tick()
        runner._agent.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_heartbeat_ok_no_notification(self, runner):
        runner._agent.ainvoke.return_value = {
            "messages": [MagicMock(content=HEARTBEAT_OK)]
        }
        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check disk usage"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        runner._agent.ainvoke.assert_awaited_once()
        runner._channels["telegram"].send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_findings_sends_notification(self, runner):
        runner._agent.ainvoke.return_value = {
            "messages": [MagicMock(content="Disk usage at 95% on /dev/sda1")]
        }
        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check disk usage"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        runner._agent.ainvoke.assert_awaited_once()
        runner._channels["telegram"].send.assert_awaited_once()
        call_args = runner._channels["telegram"].send.call_args
        assert "12345" in call_args.args
        assert "Disk usage" in call_args.args[1]

    @pytest.mark.asyncio
    async def test_tick_agent_failure_increments_counter(self, runner):
        runner._agent.ainvoke.side_effect = RuntimeError("model error")
        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check something"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        assert runner._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_tick_auto_disables_after_max_failures(self, runner):
        runner._agent.ainvoke.side_effect = RuntimeError("model error")
        runner._consecutive_failures = 2  # one away from max (3)
        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check something"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        assert runner._consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_tick_success_resets_failure_counter(self, runner):
        runner._consecutive_failures = 2
        runner._agent.ainvoke.return_value = {
            "messages": [MagicMock(content=HEARTBEAT_OK)]
        }
        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check disk"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        assert runner._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_tick_no_notify_chat_id(self):
        cfg = _config(enabled=True, notify_chat_id="")
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [MagicMock(content="Something is wrong")]
        }
        channel = AsyncMock()
        runner = HeartbeatRunner(cfg, agent, {"telegram": channel})

        with (
            patch("deepclaw.heartbeat._load_checklist", return_value="- Check something"),
            patch("deepclaw.heartbeat.is_quiet_hours", return_value=False),
        ):
            await runner._tick()

        # Should not try to send since no chat_id configured
        channel.send.assert_not_awaited()
