# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Interactive per-component hardening setup — SELinux policy, AppArmor addendum.

One ``[Y/s/n]`` flow shared by every frontend: say what state the host
is in, where the change lands, and exactly which ``sudo bash`` command
would make it — then run that command, or print the rules it applies.
``terok setup selinux`` / ``terok-executor setup selinux`` /
``terok-sandbox setup selinux`` (and the ``apparmor`` twins) all route
through [`handle_setup_component`][terok_sandbox._setup_manual.handle_setup_component],
and the aggregate setup's hints name the same verbs via
[`setup_invocation`][terok_sandbox.operator_cli.setup_invocation].

The split matters for sudo: everything that needs the operator's context
— the venv path of the bundled installer, the resolved sandbox-live root
— is resolved and rendered *before* privilege escalation, and the shown
command is the argv that runs, built from it.  Installing is therefore
always interactive (sudo asks for the password mid-flow), so there is
deliberately no ``--yes``: an unattended run copies the printed command
and executes it directly instead.

What the ``s`` answer prints is the installer's own input — the rendered
AppArmor block, the policy source file — so an operator reviews the
change itself, and can diff it against the host afterwards.
"""

from __future__ import annotations

import shlex
import subprocess  # nosec B404 — runs the bundled, audited installer via sudo
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from terok_util import find_host_tool, require_no_downgrade

from ._util._apparmor import (
    AppArmorStatus,
    check_status as _check_apparmor,
    dnsmasq_profile as _dnsmasq_profile,
    install_script_path as _apparmor_install_script,
    local_include_path as _local_include_path,
    render_addendum as _render_addendum,
    state_root as _apparmor_state_root,
)
from ._util._selinux import (
    SELINUX_MODULE_NAME,
    SelinuxStatus,
    check_status as _check_selinux,
    install_script_path as _selinux_install_script,
    policy_source_path as _policy_source_path,
)

if TYPE_CHECKING:
    from .config import SandboxConfig

#: The hardening components ``setup <component>`` can install — the single
#: roster every frontend's argument validation and hint rendering consume.
SETUP_COMPONENTS: tuple[str, ...] = ("selinux", "apparmor")

_SELINUX_STATUS_LINES = {
    SelinuxStatus.OK: f"The {SELINUX_MODULE_NAME} policy is loaded and current.",
    SelinuxStatus.POLICY_MISSING: f"The {SELINUX_MODULE_NAME} policy is not loaded.",
    SelinuxStatus.POLICY_OUTDATED: f"The loaded {SELINUX_MODULE_NAME} policy is outdated.",
    SelinuxStatus.LIBSELINUX_MISSING: "libselinux is not loadable; the policy state is unknown.",
    SelinuxStatus.NOT_APPLICABLE_TCP_MODE: (
        "Services run in TCP mode. No SELinux policy is needed on this host."
    ),
    SelinuxStatus.NOT_APPLICABLE_PERMISSIVE: (
        "SELinux is disabled or permissive. No policy is needed on this host."
    ),
}

_APPARMOR_STATUS_LINES = {
    AppArmorStatus.OK: "The terok dnsmasq addendum is installed and current.",
    AppArmorStatus.PROFILE_MISSING: (
        "dnsmasq is AppArmor-confined here; the terok addendum is not installed."
    ),
    AppArmorStatus.PROFILE_OUTDATED: "The installed terok dnsmasq addendum is an older revision.",
    AppArmorStatus.NOT_APPLICABLE: (
        "AppArmor does not confine dnsmasq on this host. Nothing to install."
    ),
}


def handle_setup_component(
    component: str | None,
    *,
    show_only: bool = False,
    cfg: SandboxConfig | None = None,
) -> int:
    """Run the interactive installer for one hardening component.

    The whole ``setup <component>`` contract lives here, so every
    frontend is one routing line: a missing or unknown component name is
    answered from here too, spelled with the caller's own invocation.
    """
    from .config import SandboxConfig

    if component == "selinux":
        comp = _selinux_component(cfg or SandboxConfig())
    elif component == "apparmor":
        comp = _apparmor_component()
    else:
        from .operator_cli import setup_invocation

        named = (
            f"unknown setup component {component!r}" if component else "--show needs a component"
        )
        raise SystemExit(f"{named}: {setup_invocation()} <{'|'.join(SETUP_COMPONENTS)}>")
    if not show_only:
        from .setup import check_setup

        require_no_downgrade(check_setup(cfg))
    return _run_component(comp, show_only=show_only)


@dataclass(frozen=True)
class _Component:
    """One hardening component the interactive flow can install."""

    status_line: str
    """One-sentence current state, printed first."""

    destination: str
    """Where the change lands, and how to undo it.  Empty when the host
    has nothing to install into — the status line already says so."""

    source: str
    """The rules the installer applies — the ``s`` / ``--show`` answer."""

    script_args: tuple[str, ...]
    """``bash`` argv tail (installer path + its arguments) for the sudo run."""

    action_needed: bool
    """Whether the status calls for an install.  The one thing it decides
    is the exit code where no install happens: "nothing to do" must not
    read as failure to a scripted caller."""

    @property
    def command(self) -> str:
        """The ``sudo bash …`` line — the argv that runs, spelled for a shell.

        Derived, so what the operator reads and what
        [`_sudo_run`][terok_sandbox._setup_manual._sudo_run] executes
        cannot drift, and a path with spaces stays copy-pasteable.
        """
        return f"sudo bash {shlex.join(self.script_args)}"


def _selinux_component(cfg: SandboxConfig) -> _Component:
    """Assemble the SELinux policy component for *cfg*'s services mode."""
    status = _check_selinux(services_mode=cfg.services_mode).status
    return _Component(
        status_line=_SELINUX_STATUS_LINES[status],
        destination=(
            f"Loads SELinux module {SELINUX_MODULE_NAME} into the system policy"
            f" (undo: sudo semodule -r {SELINUX_MODULE_NAME})"
        ),
        source=_policy_source_path().read_text(),
        script_args=(str(_selinux_install_script()),),
        action_needed=status.action_needed,
    )


def _apparmor_component() -> _Component:
    """Assemble the AppArmor addendum component for this host."""
    status = _check_apparmor().status
    root = _apparmor_state_root()
    profile = _dnsmasq_profile()
    return _Component(
        status_line=_APPARMOR_STATUS_LINES[status],
        destination=(
            f"Writes the managed block into {_local_include_path(profile)}, then reloads {profile}"
            if profile is not None
            else ""
        ),
        source=_render_addendum(root),
        script_args=(str(_apparmor_install_script()), str(root)),
        action_needed=status.action_needed,
    )


def _sudo_run(script_args: tuple[str, ...]) -> int:
    """Run ``sudo bash <installer> …`` with inherited stdio.

    Inherited stdio keeps sudo's password prompt working.  A missing
    ``sudo`` fails with a plain sentence naming the alternative — the
    printed command still works from a root shell.
    """
    sudo = find_host_tool("sudo")
    if sudo is None:
        raise SystemExit("sudo not found on PATH. Run the command above as root instead.")
    return subprocess.run([sudo, "bash", *script_args], check=False).returncode  # nosec B603


def _run_component(comp: _Component, *, show_only: bool) -> int:
    """The shared show-ask-run flow.  Returns the process exit code.

    ``--show`` prints the rules and returns.  Otherwise the host's
    state, the destination, and the exact sudo command are printed, and
    a terminal is asked to confirm (``s`` prints the rules, then the
    question comes again).  Enter installs: the verb was invoked for
    that, and both installers are idempotent, so re-installing on an
    already-current host is a no-op refresh.  Without a terminal on both
    ends only the command is printed — the install needs one for the
    sudo password.  Declining, and end-of-input, exit 0 when there was
    nothing to do and 1 when the operator left work undone; Ctrl-C exits
    130.
    """
    if show_only:
        print(comp.source, end="")
        return 0
    print(comp.status_line)
    if comp.destination:
        print(comp.destination)
    print()
    print("This will run (sudo asks for your password):")
    print(f"  {comp.command}")
    declined = 1 if comp.action_needed else 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print()
        print("Not a terminal. Run the command above directly to install.")
        return declined
    print()
    while True:
        try:
            answer = input("Proceed? [Y/s/n] (s = show the rules first): ").strip().lower()
        except EOFError:
            print()
            return declined
        except KeyboardInterrupt:
            print()
            return 130
        if answer in ("", "y", "yes"):
            print()
            return _sudo_run(comp.script_args)
        if answer == "s":
            print()
            print(comp.source, end="")
            print()
        elif answer in ("n", "no"):
            if comp.action_needed:
                print("Cancelled. You can run the command above yourself at any time.")
            return declined


__all__ = [
    "SETUP_COMPONENTS",
    "handle_setup_component",
]
