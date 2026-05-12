# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the round-trip YAML write-back helpers in ``_yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox._yaml import update_section, write_secret_text


class TestUpdateSection:
    """``update_section`` round-trips comments and merges by section."""

    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        """A missing config file is created with the section populated."""
        path = tmp_path / "config.yml"
        update_section(path, "credentials", {"use_keyring": True})
        text = path.read_text()
        assert "credentials:" in text
        assert "use_keyring: true" in text

    def test_merges_into_existing_section(self, tmp_path: Path) -> None:
        """Existing keys outside ``updates`` are preserved verbatim."""
        path = tmp_path / "config.yml"
        path.write_text(
            "# top-level comment\n"
            "credentials:\n"
            "  passphrase: secret  # inline\n"
            "  unrelated: keep-me\n"
        )
        update_section(path, "credentials", {"use_keyring": True})
        text = path.read_text()
        assert "# top-level comment" in text  # round-trip preserves the leading comment
        assert "unrelated: keep-me" in text
        assert "use_keyring: true" in text

    def test_replaces_non_dict_section(self, tmp_path: Path) -> None:
        """A stale scalar at ``data[section]`` is replaced, not ``update``-d."""
        path = tmp_path / "config.yml"
        path.write_text("credentials: legacy-string\n")
        update_section(path, "credentials", {"use_keyring": True})
        text = path.read_text()
        assert "use_keyring: true" in text
        assert "legacy-string" not in text

    def test_rejects_non_dict_root(self, tmp_path: Path) -> None:
        """A YAML file whose top-level is a scalar / list fails loudly.

        Silent coercion would destroy whatever the operator had there;
        we make them rename or fix the file by hand instead.
        """
        path = tmp_path / "config.yml"
        original = "- list-not-mapping\n"
        path.write_text(original)
        with pytest.raises(ValueError, match="config.yml.*expected a mapping"):
            update_section(path, "credentials", {"use_keyring": True})
        # File on disk is untouched — the bad input is still there for the
        # operator to inspect, not silently replaced with a fresh ``{}``.
        assert path.read_text() == original

    def test_file_is_mode_0600(self, tmp_path: Path) -> None:
        """Config files often hold passphrases — written 0600 from the kernel."""
        path = tmp_path / "config.yml"
        update_section(path, "credentials", {"passphrase": "secret"})
        assert os.stat(path).st_mode & 0o777 == 0o600


class TestWriteSecretText:
    """``write_secret_text`` is atomic, restrictive, and crash-safe."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Happy path: write, read back, mode 0o600."""
        path = tmp_path / "secret"
        write_secret_text(path, "hello\n")
        assert path.read_text() == "hello\n"
        assert os.stat(path).st_mode & 0o777 == 0o600

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        """Missing parent directories are created with 0o700."""
        path = tmp_path / "nested" / "dir" / "secret"
        write_secret_text(path, "hello")
        assert path.exists()
        assert os.stat(path.parent).st_mode & 0o777 == 0o700

    def test_replaces_existing_file_atomically(self, tmp_path: Path) -> None:
        """An existing file at the target is replaced wholesale."""
        path = tmp_path / "secret"
        path.write_text("old")
        write_secret_text(path, "new")
        assert path.read_text() == "new"

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        """If ``os.replace`` raises, the temp file is unlinked, not left behind."""
        path = tmp_path / "secret"
        with patch("terok_sandbox._yaml.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                write_secret_text(path, "hello")
        # Temp file must be cleaned up; only the absent target stays absent.
        assert not path.exists()
        # No leftover ``secret.*`` tempfiles in the directory either.
        leftover = [p.name for p in tmp_path.iterdir() if p.name.startswith("secret.")]
        assert leftover == [], f"tempfile(s) survived: {leftover}"
