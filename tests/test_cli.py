"""Tests for deepclaw.cli."""

import pytest

from deepclaw import cli


class TestDoctorCommand:
    def test_reports_config_validation_error_and_exits(self, monkeypatch, capsys):
        def _raise_config_error():
            raise ValueError("terminal.compression must be one of ['none', 'rtk']")

        monkeypatch.setattr(cli, "load_config", _raise_config_error)

        with pytest.raises(SystemExit) as exc_info:
            cli._handle_doctor_command()

        assert exc_info.value.code == 1
        stdout = capsys.readouterr().out
        assert "Configuration error:" in stdout
        assert "terminal.compression" in stdout
