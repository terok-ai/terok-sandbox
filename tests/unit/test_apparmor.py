# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for terok_sandbox._util._apparmor (dnsmasq profile detection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import _setup
from terok_sandbox._util import _apparmor
from terok_sandbox._util._apparmor import (
    AppArmorCheckResult,
    AppArmorStatus,
    check_status,
)


def _prologue_of(script: Path) -> str:
    """The shared ``_reject_unsafe`` block of a bundled sudo installer."""
    text = script.read_text()
    start = text.index("# Defence-in-depth")
    return text[start : text.index("\n}\n", start) + 3]


def _arrange(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    apparmor: bool = True,
    dnsmasq: bool = True,
    profile: bool = True,
    addendum: bool = False,
    outdated: bool = False,
) -> None:
    """Point the module's sysfs/profile probes at a tmp fixture tree.

    *addendum* installs the current-revision block; *outdated* installs one
    carrying the base marker but not the current revision (a stale install).
    """
    enabled = tmp_path / "enabled"
    enabled.write_text("Y\n" if apparmor else "N\n")
    monkeypatch.setattr(_apparmor, "_APPARMOR_ENABLED", enabled)
    monkeypatch.setattr(
        _apparmor, "find_host_tool", lambda _n: "/usr/sbin/dnsmasq" if dnsmasq else None
    )
    prof = tmp_path / "etc" / "apparmor.d" / "dnsmasq"
    if profile:
        prof.parent.mkdir(parents=True, exist_ok=True)
        prof.write_text("# dnsmasq profile\n")
        if addendum or outdated:
            # Build the markers from the module constants so bumping the
            # revision never silently rots this fixture.
            marker = (
                f"# >>> {_apparmor._ADDENDUM_MARKER} (older revision) >>>"
                if outdated
                else _apparmor._addendum_header()
            )
            local = prof.parent / "local" / "dnsmasq"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(f"{marker}\nowner x r,\n")
    monkeypatch.setattr(_apparmor, "_DNSMASQ_PROFILES", (prof,))


def test_not_applicable_without_apparmor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AppArmor disabled → NOT_APPLICABLE."""
    _arrange(monkeypatch, tmp_path, apparmor=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_not_applicable_without_dnsmasq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dnsmasq absent → NOT_APPLICABLE (the dnsmasq tier wouldn't be used)."""
    _arrange(monkeypatch, tmp_path, dnsmasq=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_not_applicable_without_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No stock dnsmasq AppArmor profile → NOT_APPLICABLE."""
    _arrange(monkeypatch, tmp_path, profile=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_profile_missing_without_addendum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AppArmor + dnsmasq + stock profile but no terok addendum → PROFILE_MISSING."""
    _arrange(monkeypatch, tmp_path, addendum=False)
    assert check_status().status is AppArmorStatus.PROFILE_MISSING


def test_ok_when_addendum_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The current-revision addendum present in the local include → OK."""
    _arrange(monkeypatch, tmp_path, addendum=True)
    assert check_status().status is AppArmorStatus.OK


def test_profile_outdated_when_addendum_is_old_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An addendum with the base marker but not the current revision → PROFILE_OUTDATED."""
    _arrange(monkeypatch, tmp_path, outdated=True)
    assert check_status().status is AppArmorStatus.PROFILE_OUTDATED


def test_profile_outdated_when_revision_is_a_superstring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A future revision like r20 must not read as the current r2.

    Whole-line matching makes this structural: the r20 header is simply a
    different line, so no token-boundary reasoning is needed.
    """
    _arrange(monkeypatch, tmp_path, addendum=True)
    local = tmp_path / "etc" / "apparmor.d" / "local" / "dnsmasq"
    header = _apparmor._addendum_header()
    revision = header.split(f"{_apparmor._ADDENDUM_MARKER} ", 1)[1].split(" ", 1)[0]
    future = header.replace(f" {revision} ", f" {revision}0 ", 1)
    local.write_text(f"{future}\nowner x r,\n")
    assert check_status().status is AppArmorStatus.PROFILE_OUTDATED


def test_header_is_the_only_revision_authority() -> None:
    """The template's header line carries the marker check_status keys on.

    One bump site: the revision lives in the template and nowhere else,
    so this pins the shape the probe relies on rather than an agreement
    between two constants.
    """
    header = _apparmor._addendum_header()
    assert header.startswith(f"# >>> {_apparmor._ADDENDUM_MARKER} ")
    assert header == _apparmor._addendum_template().splitlines()[0]


def test_installer_renders_the_template_not_its_own_rules() -> None:
    """The installer consumes the sibling template — no rules of its own.

    An ``owner`` rule or a marker line inside the script would mean two
    sources of truth again; the show option would stop being truthful.
    """
    script = _apparmor.install_script_path().read_text()
    assert "dnsmasq_addendum.template" in script
    # The strip-old-block sed keeps the base marker; the revisioned header
    # and the actual rules must live only in the template.
    assert "rwk," not in script
    assert _apparmor._addendum_header() not in script


def test_installers_reject_a_writable_ancestor(tmp_path: Path) -> None:
    """The tamper gate walks every ancestor, exempting sticky dirs.

    A writable grandparent lets another user rename the parent and
    substitute the whole tree, so checking only the immediate parent
    left the sudo input swappable; ``/tmp``-style sticky dirs must not
    trip it, since the sticky bit is exactly that rename restriction.
    """
    import subprocess

    guard = "\n".join(
        [
            _prologue_of(_apparmor.install_script_path()),
            '_red=""; _reset=""',
            '_reject_unsafe "$1" && echo ACCEPTED',
        ]
    )

    def _verdict(target: Path) -> str:
        done = subprocess.run(
            ["bash", "-c", guard, "-", str(target)], capture_output=True, text=True
        )
        return done.stdout + done.stderr

    nested = tmp_path / "parent" / "child"
    nested.mkdir(parents=True)
    target = nested / "install.sh"
    target.write_text("#!/usr/bin/env bash\n")
    # Owner-writable only, whatever umask the host gives the test user: the
    # verdict under test is the ancestor walk, not the file's own mode.
    for path in (tmp_path / "parent", nested):
        path.chmod(0o755)
    target.chmod(0o644)
    assert "ACCEPTED" in _verdict(target)

    (tmp_path / "parent").chmod(0o777)
    assert "Refusing to run" in _verdict(target)

    (tmp_path / "parent").chmod(0o1777)  # sticky: only the owner may rename
    assert "ACCEPTED" in _verdict(target)


def test_both_installers_share_one_tamper_prologue() -> None:
    """An auditor who reads one installer's refusal rules has read the other's."""
    from terok_sandbox._util._selinux import install_script_path as selinux_script

    assert _prologue_of(_apparmor.install_script_path()) == _prologue_of(selinux_script())


def test_render_addendum_substitutes_the_state_root(tmp_path: Path) -> None:
    """render_addendum swaps the token, strips a trailing slash, leaves no residue."""
    root = tmp_path / "sandbox-live"
    rendered = _apparmor.render_addendum(root)
    assert f"owner {root}/tasks/*/*/shield/dnsmasq.* rwk," in rendered
    assert "@STATE_ROOT@" not in rendered
    assert rendered == _apparmor.render_addendum(Path(str(root) + "/"))


@pytest.mark.parametrize(
    "subdir",
    [
        "sandbox-live",
        "a&b",  # bash >= 5.2 patsub_replacement: unquoted '&' becomes the pattern
        "back\\slash",  # unquoted replacement also consumes backslashes
        "with space",
    ],
)
def test_render_addendum_matches_the_shell_substitution(tmp_path: Path, subdir: str) -> None:
    """Python's rendering is byte-identical to the installer's bash rendering.

    The show option's whole promise is "this is exactly what the install
    appends" — lock the two substitution implementations together by
    running the script's own expansion over the shipped template, with
    roots exercising the patsub_replacement metacharacters.
    """
    import subprocess

    root = tmp_path / subdir
    template = _apparmor.install_script_path().parent / "dnsmasq_addendum.template"
    substitution = '"${_content//@STATE_ROOT@/"$state_root"}"'
    # The snippet below must be the script's own expansion — pin it, so
    # neither side can drift to the unquoted form (bash >= 5.2 treats an
    # unquoted replacement's '&' as the matched pattern).
    assert substitution in _apparmor.install_script_path().read_text()
    shell = subprocess.run(
        [
            "bash",
            "-c",
            f'state_root="${{2%/}}"; _content="$(<"$1")"; printf "%s\\n" {substitution}',
            "-",
            str(template),
            str(root) + "/",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert shell.stdout == _apparmor.render_addendum(root)


def test_is_apparmor_enabled_false_when_sysfs_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing sysfs node (AppArmor not in the kernel) reads as disabled, not an error."""
    monkeypatch.setattr(_apparmor, "_APPARMOR_ENABLED", tmp_path / "absent")
    assert _apparmor.is_apparmor_enabled() is False


# ── setup reporting glue (terok_sandbox._setup) ─────────────────────────


def _patch_status(monkeypatch: pytest.MonkeyPatch, status: AppArmorStatus) -> None:
    """Force ``_setup.check_apparmor_status`` to report *status*."""
    monkeypatch.setattr(_setup, "check_apparmor_status", lambda: AppArmorCheckResult(status))


def test_report_apparmor_silent_when_not_applicable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hosts with no dnsmasq AppArmor profile get no stage line at all."""
    _patch_status(monkeypatch, AppArmorStatus.NOT_APPLICABLE)
    result = _setup._report_apparmor()
    assert result.status is AppArmorStatus.NOT_APPLICABLE
    assert capsys.readouterr().out == ""


def test_report_apparmor_ok_marks_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the addendum present the stage line reports it installed."""
    _patch_status(monkeypatch, AppArmorStatus.OK)
    _setup._report_apparmor()
    out = capsys.readouterr().out
    assert "AppArmor profile" in out
    assert "installed" in out


def test_report_apparmor_missing_shows_install_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A confined-but-unpatched host gets the installer invocation in its stage line."""
    _patch_status(monkeypatch, AppArmorStatus.PROFILE_MISSING)
    _setup._report_apparmor()
    assert "terok-sandbox setup apparmor" in capsys.readouterr().out


def test_report_apparmor_outdated_flags_reinstall(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale addendum reports 'outdated' and the reinstall command, not a bare OK."""
    _patch_status(monkeypatch, AppArmorStatus.PROFILE_OUTDATED)
    _setup._report_apparmor()
    out = capsys.readouterr().out
    assert "outdated" in out
    assert "terok-sandbox setup apparmor" in out


def test_install_hint_fires_for_missing_and_outdated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The end-of-setup hint block fires for a missing OR outdated addendum, quiet otherwise."""
    for needs_work in (AppArmorStatus.PROFILE_MISSING, AppArmorStatus.PROFILE_OUTDATED):
        _setup.print_apparmor_install_hint(AppArmorCheckResult(needs_work))
        out = capsys.readouterr().out
        assert "AppArmor profile recommended" in out
        assert "terok-sandbox setup apparmor" in out

    for quiet in (AppArmorStatus.OK, AppArmorStatus.NOT_APPLICABLE):
        _setup.print_apparmor_install_hint(AppArmorCheckResult(quiet))
        assert capsys.readouterr().out == ""


def test_state_root_follows_the_sandbox_live_resolver() -> None:
    """The rules name whatever root the package resolves — one resolver, full precedence."""
    from terok_sandbox.paths import sandbox_live_root

    assert _apparmor.state_root() == sandbox_live_root()


def test_state_root_honours_the_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An overridden sandbox-live root reaches the rules (terok-parity)."""
    monkeypatch.setenv("TEROK_SANDBOX_LIVE_DIR", str(tmp_path / "custom"))
    assert _apparmor.state_root() == tmp_path / "custom"
    assert f"owner {tmp_path / 'custom'}/tasks" in _apparmor.render_addendum(_apparmor.state_root())
