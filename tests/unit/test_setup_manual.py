# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the interactive per-component hardening setup flow."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from terok_sandbox import _setup_manual
from terok_sandbox._setup_manual import (
    SETUP_COMPONENTS,
    _Component,
    _run_component,
    handle_setup_component,
)

_NEEDED = _Component(
    status_line="The widget is not installed.",
    destination="Writes /etc/widget.d/local (undo: rm it)",
    source="rule one,\nrule two,\n",
    script_args=("/somewhere/install.sh", "/some/root"),
    action_needed=True,
)

_SETTLED = _Component(
    status_line="The widget is installed and current.",
    destination="Writes /etc/widget.d/local (undo: rm it)",
    source="rule one,\nrule two,\n",
    script_args=("/somewhere/install.sh", "/some/root"),
    action_needed=False,
)


def _stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer both host probes from memory.

    The flow's inputs are the operator's config and the bundled
    resources; the host's own SELinux/AppArmor state is exactly what a
    unit test must not consult.
    """
    from terok_sandbox._util import _apparmor, _selinux

    monkeypatch.setattr(
        _setup_manual,
        "_check_apparmor",
        lambda: _apparmor.AppArmorCheckResult(_apparmor.AppArmorStatus.PROFILE_MISSING),
    )
    monkeypatch.setattr(
        _setup_manual,
        "_check_selinux",
        lambda **_kw: _selinux.SelinuxCheckResult(_selinux.SelinuxStatus.POLICY_MISSING),
    )


def _flow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    answers: list[str] | None = None,
    tty: bool = True,
    show_only: bool = False,
    component: _Component = _NEEDED,
) -> tuple[int, str, mock.MagicMock]:
    """Drive _run_component with canned answers; return (exit, stdout, sudo mock)."""
    monkeypatch.setattr(_setup_manual.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(_setup_manual.sys.stdout, "isatty", lambda: tty, raising=False)
    if answers is not None:
        answer_iter = iter(answers)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answer_iter))
    sudo = mock.MagicMock(return_value=0)
    monkeypatch.setattr(_setup_manual, "_sudo_run", sudo)
    code = _run_component(component, show_only=show_only)
    return code, capsys.readouterr().out, sudo


class TestRunComponent:
    """The show-ask-run flow: what the operator reads, and what runs."""

    def test_show_only_prints_the_source_and_returns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, sudo = _flow(monkeypatch, capsys, show_only=True)
        assert code == 0
        assert out == _NEEDED.source
        sudo.assert_not_called()

    def test_tells_state_destination_and_command_before_asking(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The whole story lands before the prompt — state, where, what, and the sudo warning."""
        _code, out, _sudo = _flow(monkeypatch, capsys, answers=["n"])
        assert _NEEDED.status_line in out
        assert _NEEDED.destination in out
        assert "sudo asks for your password" in out
        assert _NEEDED.command in out

    @pytest.mark.parametrize("answer", ["", "y", "yes"])
    def test_confirming_runs_exactly_the_shown_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str
    ) -> None:
        code, _out, sudo = _flow(monkeypatch, capsys, answers=[answer])
        assert code == 0
        sudo.assert_called_once_with(_NEEDED.script_args)

    def test_enter_installs_even_when_nothing_is_needed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The verb was invoked explicitly and the installers are idempotent."""
        code, _out, sudo = _flow(monkeypatch, capsys, answers=[""], component=_SETTLED)
        assert code == 0
        sudo.assert_called_once()

    def test_s_prints_the_rules_then_asks_again(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, sudo = _flow(monkeypatch, capsys, answers=["s", "y"])
        assert code == 0
        assert _NEEDED.source in out
        sudo.assert_called_once()

    def test_unknown_answer_asks_again(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _out, sudo = _flow(monkeypatch, capsys, answers=["what?", "y"])
        assert code == 0
        sudo.assert_called_once()

    def test_declining_leaves_work_undone(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, sudo = _flow(monkeypatch, capsys, answers=["n"])
        assert code == 1
        assert "Cancelled" in out
        sudo.assert_not_called()

    def test_declining_when_settled_is_success(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Nothing to do must not read as failure — and needs no apology."""
        code, out, sudo = _flow(monkeypatch, capsys, answers=["n"], component=_SETTLED)
        assert code == 0
        assert "Cancelled" not in out
        sudo.assert_not_called()

    @pytest.mark.parametrize(
        ("component", "expected"), [(_NEEDED, 1), (_SETTLED, 0)], ids=["needed", "settled"]
    )
    def test_without_a_terminal_prints_the_command_and_declines(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        component: _Component,
        expected: int,
    ) -> None:
        code, out, sudo = _flow(monkeypatch, capsys, tty=False, component=component)
        assert code == expected
        assert component.command in out
        assert "Run the command above directly" in out
        sudo.assert_not_called()

    def test_end_of_input_declines(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_setup_manual.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(_setup_manual.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", mock.MagicMock(side_effect=EOFError))
        monkeypatch.setattr(_setup_manual, "_sudo_run", mock.MagicMock())
        assert _run_component(_NEEDED, show_only=False) == 1
        assert _run_component(_SETTLED, show_only=False) == 0

    def test_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_setup_manual.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(_setup_manual.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", mock.MagicMock(side_effect=KeyboardInterrupt))
        assert _run_component(_NEEDED, show_only=False) == 130

    def test_installer_exit_code_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_setup_manual.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(_setup_manual.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        monkeypatch.setattr(_setup_manual, "_sudo_run", mock.MagicMock(return_value=7))
        assert _run_component(_NEEDED, show_only=False) == 7


class TestShownCommand:
    """The shown command is the argv that runs — derived, never restated."""

    def test_names_the_installer_and_its_arguments(self) -> None:
        assert _NEEDED.command == "sudo bash /somewhere/install.sh /some/root"

    def test_quotes_paths_a_shell_would_split(self) -> None:
        """A venv under a path with spaces stays copy-pasteable."""
        comp = _Component(
            status_line="",
            destination="",
            source="",
            script_args=("/opt/my venv/install.sh", "/home/a b/live"),
            action_needed=True,
        )
        assert comp.command == "sudo bash '/opt/my venv/install.sh' '/home/a b/live'"


class TestHandleSetupComponent:
    """The one door: routing, and the messages for no/unknown component."""

    def test_unknown_component_names_the_choices_and_the_invocation(self) -> None:
        with pytest.raises(SystemExit, match=r"unknown setup component 'bogus'") as exc_info:
            handle_setup_component("bogus")
        assert "terok-sandbox setup <selinux|apparmor>" in str(exc_info.value.code)

    def test_show_without_a_component_says_so(self) -> None:
        with pytest.raises(SystemExit, match="--show needs a component") as exc_info:
            handle_setup_component(None, show_only=True)
        assert "terok-sandbox setup <selinux|apparmor>" in str(exc_info.value.code)

    def test_message_follows_the_declared_invocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from terok_sandbox.operator_cli import SETUP_INVOCATION_ENV

        monkeypatch.setenv(SETUP_INVOCATION_ENV, "terok setup")
        with pytest.raises(SystemExit, match="terok setup <selinux|apparmor>"):
            handle_setup_component(None)

    @pytest.mark.parametrize("component", SETUP_COMPONENTS)
    def test_every_component_in_the_roster_routes(
        self, monkeypatch: pytest.MonkeyPatch, component: str
    ) -> None:
        """Every roster entry reaches a builder — no host probes involved."""
        _stub_probes(monkeypatch)
        seen: list[_Component] = []
        monkeypatch.setattr(
            _setup_manual, "_run_component", lambda comp, *, show_only: seen.append(comp) or 0
        )
        assert handle_setup_component(component, show_only=True) == 0
        assert seen[0].source.strip(), "a component must have rules to show"


class TestComponents:
    """Each component names its own state, destination, rules, and argv."""

    def test_apparmor_speaks_about_the_resolved_root_throughout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One resolver: the command, the rules, and the argv all name it."""
        from terok_sandbox._util import _apparmor

        root = tmp_path / "custom-live"
        monkeypatch.setenv("TEROK_SANDBOX_LIVE_DIR", str(root))
        monkeypatch.setattr(
            _setup_manual,
            "_check_apparmor",
            lambda: _apparmor.AppArmorCheckResult(_apparmor.AppArmorStatus.PROFILE_MISSING),
        )
        comp = _setup_manual._apparmor_component()
        assert comp.script_args[-1] == str(root)
        assert str(root) in comp.command
        assert f"owner {root}/tasks" in comp.source
        assert comp.action_needed is True

    def test_apparmor_destination_names_the_local_include(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from terok_sandbox._util import _apparmor

        profile = tmp_path / "etc" / "apparmor.d" / "usr.sbin.dnsmasq"
        monkeypatch.setattr(_setup_manual, "_dnsmasq_profile", lambda: profile)
        monkeypatch.setattr(
            _setup_manual,
            "_check_apparmor",
            lambda: _apparmor.AppArmorCheckResult(_apparmor.AppArmorStatus.PROFILE_MISSING),
        )
        comp = _setup_manual._apparmor_component()
        assert str(profile.parent / "local" / profile.name) in comp.destination
        assert str(profile) in comp.destination

    def test_apparmor_destination_empty_without_a_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing to write into — the status line already explains why."""
        from terok_sandbox._util import _apparmor

        monkeypatch.setattr(_setup_manual, "_dnsmasq_profile", lambda: None)
        monkeypatch.setattr(
            _setup_manual,
            "_check_apparmor",
            lambda: _apparmor.AppArmorCheckResult(_apparmor.AppArmorStatus.NOT_APPLICABLE),
        )
        comp = _setup_manual._apparmor_component()
        assert comp.destination == ""
        assert comp.action_needed is False

    def test_selinux_uses_the_cfg_services_mode_and_names_the_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from terok_sandbox._util import _selinux

        seen: dict[str, object] = {}

        def _fake_check(*, services_mode: object) -> _selinux.SelinuxCheckResult:
            seen["mode"] = services_mode
            return _selinux.SelinuxCheckResult(_selinux.SelinuxStatus.POLICY_MISSING)

        monkeypatch.setattr(_setup_manual, "_check_selinux", _fake_check)
        comp = _setup_manual._selinux_component(mock.MagicMock(services_mode="socket"))
        assert seen["mode"] == "socket"
        assert _selinux.SELINUX_MODULE_NAME in comp.destination
        assert f"semodule -r {_selinux.SELINUX_MODULE_NAME}" in comp.destination
        assert comp.action_needed is True

    def test_selinux_source_is_the_policy_file_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewed text and the compiler's input are the same bytes."""
        from terok_sandbox._util._selinux import policy_source_path

        _stub_probes(monkeypatch)
        comp = _setup_manual._selinux_component(mock.MagicMock(services_mode="tcp"))
        assert comp.source == policy_source_path().read_text()
        assert "module terok_socket" in comp.source
        assert "#" in comp.source, "the rationale comments are part of what the operator reviews"


@pytest.mark.parametrize("owner", ["terok-sandbox", "terok-shield"])
@pytest.mark.parametrize("component", ["selinux", "apparmor"])
def test_manual_install_refuses_dependency_closure_downgrades(owner, component, monkeypatch):
    """Every public mutation route checks the full closure before an installer runs."""
    from terok_util import SetupCheck, SetupDowngradeError, SetupStatus

    monkeypatch.setattr(_setup_manual, "_selinux_component", lambda _: _NEEDED)
    monkeypatch.setattr(_setup_manual, "_apparmor_component", lambda: _NEEDED)
    check = mock.Mock(return_value=(SetupCheck(owner, "receipt", SetupStatus.DOWNGRADE),))
    monkeypatch.setattr("terok_sandbox.setup.check_setup", check)
    run = mock.Mock()
    monkeypatch.setattr(_setup_manual, "_run_component", run)
    with pytest.raises(SetupDowngradeError):
        handle_setup_component(component)
    run.assert_not_called()


@pytest.mark.parametrize("component", ["selinux", "apparmor"])
def test_show_rules_remains_read_only_after_downgrade(component, monkeypatch):
    """Printing bundled rules doesn't need permission to overwrite installed setup."""
    monkeypatch.setattr(_setup_manual, "_selinux_component", lambda _: _NEEDED)
    monkeypatch.setattr(_setup_manual, "_apparmor_component", lambda: _NEEDED)
    check = mock.Mock(side_effect=AssertionError("read-only display checked setup"))
    monkeypatch.setattr("terok_sandbox.setup.check_setup", check)
    assert handle_setup_component(component, show_only=True) == 0
    check.assert_not_called()
