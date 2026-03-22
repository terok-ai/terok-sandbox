# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests verifying the package imports correctly."""


def test_import_package():
    """Package root is importable and exposes __version__."""
    import terok_sandbox

    assert hasattr(terok_sandbox, "__version__")
    assert isinstance(terok_sandbox.__version__, str)


def test_import_util():
    """Vendored _util subpackage is importable."""
    from terok_sandbox._util import dump, ensure_dir, ensure_dir_writable, load, render_template

    assert callable(ensure_dir)
    assert callable(ensure_dir_writable)
    assert callable(load)
    assert callable(dump)
    assert callable(render_template)


def test_import_gate():
    """Gate subpackage is importable."""
    from terok_sandbox.gate import server

    assert hasattr(server, "main")
    assert hasattr(server, "TokenStore")


def test_cli_main_exits_cleanly():
    """CLI entry point prints version and exits cleanly."""
    import pytest

    from terok_sandbox.cli import main

    with pytest.raises(SystemExit, match="0"):
        main()
