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
    RawCredentialsSection,
    RawHooksSection,
    RawNetworkSection,
    RawPathsSection,
    RawRunSection,
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


def test_credentials_section_rejects_removed_plaintext_passphrase() -> None:
    """The removed plaintext tier fails with migration directions, not pydantic noise."""
    with pytest.raises(ValidationError, match="passphrase_command: cat /path/to/that/file"):
        RawCredentialsSection.model_validate({"passphrase": "hunter2"})


def test_credentials_section_accepts_explicit_null_passphrase() -> None:
    """``passphrase: null`` (a commented-out-then-nulled config) stays valid."""
    section = RawCredentialsSection.model_validate({"passphrase": None})
    assert section.passphrase is None


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


def test_run_section_defaults_runtime_to_none_so_layers_can_distinguish_unset() -> None:
    """``runtime: None`` (not ``"crun"``) at the field level lets the orchestrator
    tell "operator didn't set it" from "operator picked crun explicitly" — the
    distinction matters when project values inherit/override global values.
    """
    section = RawRunSection.model_validate({})
    assert section.runtime is None
    assert section.shutdown_timeout == 10
    assert section.nested_containers is False
    assert section.hooks == RawHooksSection()


def test_run_section_rejects_legacy_podman_value() -> None:
    """``"podman"`` was the v0 value name and is now invalid.  Hard error
    rather than silent acceptance — the new value is ``"crun"`` (the
    actual OCI runtime podman drives by default).
    """
    with pytest.raises(ValidationError, match="runtime"):
        RawRunSection.model_validate({"runtime": "podman"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("all", "all"), ("nvidia,amd", "nvidia,amd"), (None, None)],
    ids=repr,
)
def test_run_section_gpus_accepts_selector_shapes(value: object, expected: object) -> None:
    """Booleans, selector strings, and None pass through unchanged."""
    assert RawRunSection.model_validate({"gpus": value}).gpus == expected


@pytest.mark.parametrize("value", [0, 1, 1.0], ids=repr)
def test_run_section_gpus_rejects_numeric(value: object) -> None:
    """``gpus: 1`` must not lax-coerce into ``True`` and enable every GPU."""
    with pytest.raises(ValidationError, match="gpus"):
        RawRunSection.model_validate({"gpus": value})


def test_run_section_gpus_rejects_unknown_vendor() -> None:
    """Unknown vendor tokens fail at parse time — even alongside ``all``."""
    with pytest.raises(ValidationError, match="matrox"):
        RawRunSection.model_validate({"gpus": "all,matrox"})


def test_run_section_blank_memory_cpus_normalised_to_none() -> None:
    """An accidentally-empty ``memory: ""`` in YAML reads as None rather
    than as the literal empty string, so podman doesn't receive a bare
    ``--memory ""`` flag."""
    section = RawRunSection.model_validate({"memory": "  ", "cpus": ""})
    assert section.memory is None
    assert section.cpus is None


@pytest.mark.parametrize(
    "value",
    [
        "4g",
        "512m",
        "256K",
        "1.5G",
        "1024",
        "2B",
        "0",
        "4gb",  # podman accepts the redundant trailing 'b'
        "4gib",  # binary-IEC suffix
        "4GiB",  # case-insensitive across the board
        "4 g",  # podman tolerates one space between number and unit
    ],
)
def test_run_section_memory_accepts_podman_grammar(value: str) -> None:
    """Mirrors ``docker/go-units.sizeRegex`` (what podman's ``--memory`` accepts).

    ``"0"`` is format-valid; semantics are podman's call (the validator
    is format-only — see [`RawRunSection`][terok_sandbox.config_schema.RawRunSection]).
    """
    assert RawRunSection.model_validate({"memory": value}).memory == value


@pytest.mark.parametrize("value", ["two", "-1g", "4g ", "4.g", ".5g", "4  g", ""])
def test_run_section_memory_rejects_malformed(value: str) -> None:
    """Malformed values fail at parse time, not task launch.

    Blank → ``None`` via ``_blank_to_none``; listed only as a marker
    that the format check fires for non-blank inputs.  Leading-dot
    decimals (``".5g"``) are rejected — the regex requires a leading
    digit, so the canonical form is ``"0.5g"`` / ``"512m"``.  Trailing
    whitespace and double spaces are rejected (only the single
    podman-tolerated space between number and unit is accepted).
    """
    if value == "":
        # blank → None via the upstream coercion, not a validator failure
        assert RawRunSection.model_validate({"memory": value}).memory is None
        return
    with pytest.raises(ValidationError, match="memory"):
        RawRunSection.model_validate({"memory": value})


@pytest.mark.parametrize("value", ["2", "2.0", "0.5", "16", "0"])
def test_run_section_cpus_accepts_decimals(value: str) -> None:
    """Non-negative decimal; ``"0"`` is format-valid (same format-only
    contract as memory)."""
    assert RawRunSection.model_validate({"cpus": value}).cpus == value


@pytest.mark.parametrize(
    "raw,expected",
    [(2, "2"), (4, "4"), (0, "0"), (0.5, "0.5"), (1024, "1024")],
)
def test_run_section_accepts_numeric_yaml_input(raw: object, expected: str) -> None:
    """``cpus: 2`` / ``memory: 1024`` (YAML int/float) coerce to str.

    Without this, a perfectly valid project.yml would fail with a
    confusing "Input should be a valid string" pydantic error.
    """
    assert RawRunSection.model_validate({"cpus": raw}).cpus == expected
    assert RawRunSection.model_validate({"memory": raw}).memory == expected


@pytest.mark.parametrize("raw", [True, False])
def test_run_section_rejects_bool(raw: bool) -> None:
    """``bool`` is an ``int`` subclass — reject explicitly so ``cpus: true``
    doesn't silently coerce to ``"True"`` and then fail the format check
    with a less helpful message."""
    with pytest.raises(ValidationError):
        RawRunSection.model_validate({"cpus": raw})
    with pytest.raises(ValidationError):
        RawRunSection.model_validate({"memory": raw})


@pytest.mark.parametrize("value", ["two", "-1", "1.5x", "1,5", "2.0 ", ".5"])
def test_run_section_cpus_rejects_malformed(value: str) -> None:
    """``".5"`` is rejected for the same leading-digit reason as
    ``".5g"`` — canonical form is ``"0.5"``."""
    with pytest.raises(ValidationError, match="cpus"):
        RawRunSection.model_validate({"cpus": value})


def test_run_section_none_hooks_becomes_empty_subsection() -> None:
    """``hooks:`` written with no value (YAML null) shouldn't crash — coerce
    to empty defaults so all four hook fields stay ``None``.
    """
    section = RawRunSection.model_validate({"hooks": None})
    assert section.hooks.pre_start is None
    assert section.hooks.post_start is None
    assert section.hooks.post_ready is None
    assert section.hooks.post_stop is None


def test_run_section_rejects_typo_in_owned_key() -> None:
    """Sandbox-owned strictness — a misspelled key surfaces loudly."""
    with pytest.raises(ValidationError, match="shutdwon_timeout"):
        RawRunSection.model_validate({"shutdwon_timeout": 5})


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

    This is the contract [`terok_executor.config_schema.ExecutorConfigView`][terok_executor.config_schema.ExecutorConfigView]
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
