# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Storage hints require independent evidence and never replace a launch error."""

from __future__ import annotations

import errno
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from terok_sandbox import PodmanRuntime, Sandbox
from terok_sandbox.runtime.storage_diagnostics import storage_permission_hint
from tests.constants import MOCK_HOST_BIN


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A private storage parent whose group falls outside the rootless mapping."""
    parent = tmp_path / "private"
    root = parent / "storage"
    root.mkdir(parents=True)
    parent.chmod(0o700)
    root.chmod(0o755)
    uid = parent.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: uid or 1000)
    if uid == 0:
        pytest.skip("the evidence test needs a non-root-owned directory")
    mapped_gid = parent.stat().st_gid
    real_stat = Path.stat

    def directory_metadata(path: Path, **kwargs: object) -> os.stat_result:
        metadata = real_stat(path, **kwargs)
        if path == parent:
            fields = list(metadata)
            fields[5] = mapped_gid + 1
            return os.stat_result(fields)
        return metadata

    monkeypatch.setattr(Path, "stat", directory_metadata)
    info = {
        "host": {
            "security": {"rootless": True},
            "serviceIsRemote": False,
            "idMappings": {"gidmap": [{"host_id": mapped_gid, "size": 1}]},
        },
        "store": {"graphRoot": str(root)},
    }
    mappings = {"UidMap": ["0:1:1000", "1000:0:1", "1001:1001:64536"]}
    responses = [info, mappings]
    run = Mock(
        side_effect=lambda *a, **kw: subprocess.CompletedProcess(a, 0, json.dumps(responses.pop(0)))
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        "terok_sandbox.runtime.storage_diagnostics.require_host_tool",
        lambda name: str(MOCK_HOST_BIN / name),
    )
    acl = Mock(side_effect=OSError(errno.ENODATA, "no ACL"))
    monkeypatch.setattr(os, "getxattr", acl)
    error = (
        "Error: runc: runc create failed: unable to start container process: "
        "error during container init: error preparing rootfs: remount-propagation "
        f"dst={root}/overlay/layer/merged, flags=MS_PRIVATE: permission denied: OCI permission denied"
    )
    return {
        "parent": parent,
        "root": root,
        "info": info,
        "mappings": mappings,
        "run": run,
        "acl": acl,
        "error": error,
    }


def test_matching_failure_gets_local_nonrepair_hint(storage: dict) -> None:
    before = storage["parent"].stat()
    hint = storage_permission_hint(storage["error"], "task")
    assert "Hint:" in hint
    assert str(storage["parent"]) in hint
    assert "Do not recursively chown/chmod" in hint
    assert storage["parent"].stat() == before
    assert storage["run"].call_count == 2
    assert all(call.kwargs["timeout"] == 3 for call in storage["run"].call_args_list)


@pytest.mark.parametrize(
    "case",
    [
        "unrelated",
        "rootful",
        "remote",
        "mapped",
        "traversable",
        "plain-root",
        "acl",
        "acl-unknown",
        "empty-map",
        "invalid-gid-map",
        "relative-store",
        "outside-store",
    ],
)
def test_ambiguous_or_valid_configurations_get_no_hint(storage: dict, case: str) -> None:
    error = storage["error"]
    match case:
        case "unrelated":
            error = "permission denied opening an unrelated file"
        case "rootful":
            storage["info"]["host"]["security"]["rootless"] = False
        case "remote":
            storage["info"]["host"]["serviceIsRemote"] = True
        case "mapped":
            storage["info"]["host"]["idMappings"]["gidmap"].append(
                {"host_id": storage["parent"].stat().st_gid, "size": 1}
            )
        case "traversable":
            storage["parent"].chmod(0o701)
        case "plain-root":
            storage["mappings"]["UidMap"] = ["0:0:65537"]
        case "acl":
            storage["acl"].side_effect = None
            storage["acl"].return_value = b"an access ACL"
        case "acl-unknown":
            storage["acl"].side_effect = OSError(errno.EOPNOTSUPP, "unknown ACL support")
        case "empty-map":
            storage["info"]["host"]["idMappings"]["gidmap"] = []
        case "invalid-gid-map":
            storage["info"]["host"]["idMappings"]["gidmap"] = [{"host_id": -1, "size": 1}]
        case "relative-store":
            storage["info"]["store"]["graphRoot"] = storage["root"].name
        case "outside-store":
            different = storage["root"] / "different"
            different.mkdir()
            storage["info"]["store"]["graphRoot"] = str(different)
    assert storage_permission_hint(error, "task") == ""
    if case == "unrelated":
        storage["run"].assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("podman missing"),
        subprocess.TimeoutExpired("podman info", 3),
        subprocess.CalledProcessError(125, "podman info"),
        ValueError("invalid JSON"),
    ],
)
def test_failed_probe_leaves_original_error_alone(storage: dict, failure: Exception) -> None:
    storage["run"].side_effect = failure
    assert storage_permission_hint(storage["error"], "task") == ""


@pytest.mark.parametrize("mapping", [None, [], {}, [None], ["bad"], ["0:1:0"]])
def test_unknown_container_mapping_has_no_hint(storage: dict, mapping: object) -> None:
    storage["mappings"]["UidMap"] = mapping
    assert storage_permission_hint(storage["error"], "task") == ""


def test_symlinked_storage_checks_resolved_ancestors(storage: dict, tmp_path: Path) -> None:
    link = tmp_path / "store-link"
    link.symlink_to(storage["root"], target_is_directory=True)
    storage["info"]["store"]["graphRoot"] = str(link)
    error = storage["error"].replace(str(storage["root"]), str(link))
    assert str(storage["parent"]) in storage_permission_hint(error, "task")


def test_launch_preserves_original_error_and_cause(storage: dict) -> None:
    failure = subprocess.CalledProcessError(126, "podman run", stderr=storage["error"].encode())
    storage["run"].side_effect = [
        failure,
        subprocess.CompletedProcess([], 0, json.dumps(storage["info"])),
        subprocess.CompletedProcess([], 0, json.dumps(storage["mappings"])),
    ]
    with pytest.raises(SystemExit) as caught:
        Sandbox()._exec_podman(["podman", "run", "--name", "task"])
    assert str(caught.value).startswith("Container launch failed:\n" + storage["error"])
    assert "Hint:" in str(caught.value)
    assert caught.value.__cause__ is failure


def test_start_preserves_original_error_and_exit_status(storage: dict) -> None:
    storage["run"].side_effect = [
        subprocess.CompletedProcess([], 126, stderr=storage["error"]),
        subprocess.CompletedProcess([], 0, json.dumps(storage["info"])),
        subprocess.CompletedProcess([], 0, json.dumps(storage["mappings"])),
    ]
    with pytest.raises(RuntimeError) as caught:
        PodmanRuntime().container("task").start()
    assert "rc=126" in str(caught.value)
    assert storage["error"] in str(caught.value)
    assert "Hint:" in str(caught.value)
