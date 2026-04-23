# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sandbox-owned slice of ``config.yml``.

Two layers of validation matter here, and the tests pin both:

1. **Owned sub-sections are strict.** A typo inside a section that
   sandbox owns (e.g. ``paths.rooot``) must surface as a pydantic error
   so users notice — sandbox is the source of truth for these keys.
2. **Top-level is tolerant.** Unknown sections (terok's ``tui:``,
   executor's ``image:``, future-package keys we haven't seen yet)
   pass through silently when sandbox is run standalone.  The
   alternative — vendoring a list of foreign keys into sandbox — would
   invert the dep hierarchy.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terok_sandbox.config_schema import (
    RawNetworkSection,
    RawPathsSection,
    RawShieldSection,
    RawSSHSection,
    RawVaultSection,
    SandboxConfigView,
)

# ── Owned sub-section strictness ──────────────────────────────────────


def test_paths_section_rejects_typo_in_owned_key() -> None:
    """``extra="forbid"`` on owned sections turns ``paths.rooot`` into a clear error."""
    with pytest.raises(ValidationError) as excinfo:
        RawPathsSection.model_validate({"rooot": "/tmp"})
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_paths_section_accepts_known_keys() -> None:
    """Documented keys validate cleanly — no false-positive rejections."""
    section = RawPathsSection.model_validate(
        {"root": "/v", "build_dir": "/b", "user_projects_dir": "/p"}
    )
    assert section.root == "/v"
    assert section.build_dir == "/b"


def test_network_section_rejects_inverted_port_range() -> None:
    """The model-level validator catches ``start > end`` instead of letting it through."""
    with pytest.raises(ValidationError, match="port_range_start must be <= port_range_end"):
        RawNetworkSection.model_validate({"port_range_start": 32700, "port_range_end": 18700})


def test_vault_section_rejects_out_of_range_port() -> None:
    """Pydantic's numeric constraint surfaces a port outside ``[1, 65535]``."""
    with pytest.raises(ValidationError):
        RawVaultSection.model_validate({"port": 70000})


def test_shield_section_rejects_unknown_on_task_restart() -> None:
    """``Literal["retain", "up"]`` keeps typos like ``"restart"`` out."""
    with pytest.raises(ValidationError):
        RawShieldSection.model_validate({"on_task_restart": "restart"})


def test_ssh_section_use_personal_defaults_to_none() -> None:
    """``None`` (not False) is the default so layered configs can distinguish unset."""
    section = RawSSHSection.model_validate({})
    assert section.use_personal is None


# ── Top-level tolerance for foreign sections ──────────────────────────


def test_view_tolerates_unknown_top_level_sections() -> None:
    """Foreign sections (terok's ``tui:``, executor's ``image:``) pass through silently.

    Without this, every standalone sandbox invocation would crash the
    moment it sees a complete ecosystem config — and we'd be forced to
    vendor a list of foreign keys into sandbox to keep ``extra="forbid"``
    tractable.  ``extra="allow"`` is what makes ownership-by-package
    work without that vendoring.
    """
    raw = {
        "paths": {"root": "/v"},
        "tui": {"default_tmux": True},  # terok-owned
        "image": {"base_image": "ubuntu:24.04"},  # executor-owned
        "some_future_package": {"foo": "bar"},  # not even in v0
    }
    view = SandboxConfigView.model_validate(raw)
    assert view.paths.root == "/v"


def test_view_strictness_propagates_into_owned_sections() -> None:
    """Top-level allow doesn't leak into owned sections — typos inside still error."""
    raw = {"paths": {"rooot": "/tmp"}}  # typo inside an owned section
    with pytest.raises(ValidationError) as excinfo:
        SandboxConfigView.model_validate(raw)
    assert "rooot" in str(excinfo.value)


def test_view_default_construction_is_empty() -> None:
    """An empty ``config.yml`` is valid — every field uses safe defaults."""
    view = SandboxConfigView.model_validate({})
    assert view.paths.root is None
    assert view.shield.audit is True
    assert view.services.mode == "socket"
    assert view.network.port_range_start == 18700


# ── Composition story (subclass inherits + tolerates owned-section validators) ──


def test_subclass_can_compose_extra_sections_and_inherit_strictness() -> None:
    """A higher-level package can subclass + add fields; sandbox validators still fire.

    This is the contract :class:`terok_executor.config_schema.ExecutorConfigView`
    and terok's ``RawGlobalConfig`` rely on — sandbox-section strictness must
    survive composition.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class _ToySection(BaseModel):
        model_config = ConfigDict(extra="forbid")
        toy_value: int = 0

    class _Composed(SandboxConfigView):
        model_config = ConfigDict(extra="allow")
        toy: _ToySection = Field(default_factory=_ToySection)

    # Sandbox-owned strictness still fires through the subclass.
    with pytest.raises(ValidationError, match="rooot"):
        _Composed.model_validate({"paths": {"rooot": "/tmp"}})

    # The added field validates on its own keys.
    composed = _Composed.model_validate({"toy": {"toy_value": 42}, "paths": {"root": "/v"}})
    assert composed.toy.toy_value == 42
    assert composed.paths.root == "/v"


# ── gate_use_personal_ssh_default reader ──────────────────────────────


def test_gate_use_personal_ssh_default_returns_false_when_unset(monkeypatch) -> None:
    """A missing or empty ``ssh:`` section collapses to ``False`` — the safe default."""
    from terok_sandbox.config_schema import gate_use_personal_ssh_default

    monkeypatch.setattr(
        "terok_sandbox.config_schema.read_config_section",
        lambda _section: {},
        raising=False,
    )
    # The patch above doesn't take if the symbol isn't already a module-level
    # attribute; the function imports lazily.  Patch the source module instead.
    import terok_sandbox.paths as _paths

    monkeypatch.setattr(_paths, "read_config_section", lambda _section: {})
    assert gate_use_personal_ssh_default() is False


def test_gate_use_personal_ssh_default_reads_true_from_config(monkeypatch) -> None:
    """``ssh: {use_personal: true}`` in config.yml flips the default to True."""
    import terok_sandbox.paths as _paths
    from terok_sandbox.config_schema import gate_use_personal_ssh_default

    monkeypatch.setattr(_paths, "read_config_section", lambda _section: {"use_personal": "true"})
    # ``read_config_section`` returns ``dict[str, str]`` (everything stringified
    # before merge); pydantic coerces the string to bool.  Verifying both that
    # the wiring works and the coercion holds.
    assert gate_use_personal_ssh_default() is True


def test_gate_use_personal_ssh_default_swallows_malformed_section(monkeypatch) -> None:
    """A typo inside ``ssh:`` doesn't crash the gate — falls back to False.

    The reader is called by gate-sync paths that must be robust to a
    partially-broken config; surfacing a pydantic error here would mean
    a single bad key in the global config takes down ``terok task start``
    on every project.  The strict validation path lives in terok's
    top-level ``RawGlobalConfig`` (``extra="forbid"``) — by the time
    the gate runs, terok would already have rejected the file there.
    """
    import terok_sandbox.paths as _paths
    from terok_sandbox.config_schema import gate_use_personal_ssh_default

    monkeypatch.setattr(_paths, "read_config_section", lambda _section: {"use_persoanl": "true"})
    assert gate_use_personal_ssh_default() is False
