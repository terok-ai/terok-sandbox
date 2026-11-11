# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The optional policy shares the audited installer's validation and compile path."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from terok_sandbox._util._selinux import (
    install_script_path,
    wayland_direct_policy_source_path,
    wayland_policy_source_path,
)

_DEVEL_MAKEFILE = Path("/usr/share/selinux/devel/Makefile")
_CONTAINER_INTERFACE = _DEVEL_MAKEFILE.parent / "include/services/container.if"


@pytest.fixture
def installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Copy installer inputs and substitute tools that record, but never install, policy."""
    resources = tmp_path / "resources"
    resources.mkdir()
    for name in (
        "install_policy.sh",
        "terok_socket.te",
        "terok_wayland.te",
        "terok_wayland_direct.te",
    ):
        shutil.copyfile(install_script_path().parent / name, resources / name)
    makefile = resources / _DEVEL_MAKEFILE.name
    interface = resources / _CONTAINER_INTERFACE.relative_to(_DEVEL_MAKEFILE.parent)
    interface.parent.mkdir(parents=True)
    makefile.touch()
    interface.touch()
    script = resources / "install_policy.sh"
    script.write_text(
        script.read_text().replace(
            f"devel_makefile={_DEVEL_MAKEFILE}", f"devel_makefile={shlex.quote(str(makefile))}"
        )
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    trace = tmp_path / "trace"
    monkeypatch.setenv("TRACE", str(trace))
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for tool in ("checkmodule", "semodule_package", "semodule", "make", "m4"):
        program = binaries / tool
        program.write_text(
            "#!/bin/bash\n"
            f"printf '%s\\n' '{tool}' \"$@\" >> \"$TRACE\"\n"
            'if [[ "$1" == "-l" ]]; then '
            'printf "terok_socket 1.1\\nterok_wayland 1.1\\nterok_wayland_direct 1.1\\n"; fi\n'
        )
        program.chmod(0o700)
    return script, trace


@pytest.mark.parametrize("with_wayland", [False, True])
@pytest.mark.parametrize("with_direct", [False, True])
def test_selected_modules_compile_then_install_together(
    installer, with_wayland: bool, with_direct: bool
) -> None:
    """Independent flags add either or both modules; neither removes existing policy."""
    script, trace = installer
    flags = ["--with-wayland"] if with_wayland else []
    if with_direct:
        flags.append("--with-wayland-direct")
    result = subprocess.run(["bash", str(script), *flags], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = trace.read_text()
    assert "terok_socket.te" in calls
    assert ("terok_wayland.te" in calls) is with_wayland
    assert ("terok_wayland_direct.pp" in calls) is with_direct
    assert ("\nmake\n" in calls) is with_direct
    assert calls.count("\n-i\n") == 1
    assert "\n-r\n" not in calls
    if with_wayland:
        assert calls.index("terok_wayland.te") < calls.index("\n-i\n")
        assert "terok_wayland.pp" in calls


def test_unknown_installer_option_is_rejected_before_tools_run(installer) -> None:
    script, trace = installer
    result = subprocess.run(["bash", str(script), "--bogus"], text=True, capture_output=True)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not trace.exists()


@pytest.mark.parametrize(
    "module,flag",
    [
        ("terok_wayland", "--with-wayland"),
        ("terok_wayland_direct", "--with-wayland-direct"),
    ],
)
def test_optional_source_gets_same_symlink_protection(installer, module: str, flag: str) -> None:
    script, trace = installer
    source = script.parent / f"{module}.te"
    source.unlink()
    source.symlink_to(script.parent / "terok_socket.te")
    result = subprocess.run(["bash", str(script), flag], text=True, capture_output=True)
    assert result.returncode == 1
    assert "is a symlink" in result.stderr
    assert not trace.exists()


def test_installer_runs_through_safe_venv_directory_alias(installer) -> None:
    """A lib64-style alias no longer prevents installation before any tools run."""
    script, trace = installer
    alias = script.parent.parent / "lib64"
    alias.symlink_to(script.parent.name, target_is_directory=True)
    result = subprocess.run(["bash", str(alias / script.name)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "\n-i\n" in trace.read_text()


def test_direct_build_consumes_only_validated_source_in_private_directory(installer) -> None:
    """Refpolicy make must not discover user-controlled sibling interface files."""
    script, trace = installer
    (script.parent / "terok_wayland_direct.if").write_text("unvalidated input\n")
    make = script.parent.parent / "bin" / "make"
    make.write_text(
        "#!/bin/bash\nset -eu\n"
        '[[ "$1" == -C && "$3" == -f && "$5" == terok_wayland_direct.pp ]]\n'
        '[[ "$(stat -c %a "$2")" == 700 ]]\n'
        '[[ -f "$2/terok_wayland_direct.te" ]]\n'
        '[[ ! -e "$2/terok_wayland_direct.if" ]]\n'
        'printf "%s\\n" "$2" > "$TRACE.builddir"\n'
    )
    result = subprocess.run(
        ["bash", str(script), "--with-wayland-direct"], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    builddir = Path(trace.with_suffix(".builddir").read_text().strip())
    assert builddir.parent == script.parent.parent
    assert not builddir.exists()
    assert "\n-i\n" in trace.read_text()


def test_failed_direct_build_never_installs_any_module(installer) -> None:
    """A failed optional build cannot partially install the selected policy set."""
    script, trace = installer
    make = script.parent.parent / "bin" / "make"
    make.write_text("#!/bin/bash\nexit 7\n")
    result = subprocess.run(
        ["bash", str(script), "--with-wayland-direct"], text=True, capture_output=True
    )
    assert result.returncode == 7
    assert "\n-i\n" not in trace.read_text()


def test_missing_direct_build_tool_fails_before_compiling(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct-only build dependency is checked before producing any policy package."""
    script, trace = installer
    bash = shutil.which("bash")
    binaries = script.parent.parent / "bin"
    (binaries / "m4").unlink()
    for tool in ("dirname", "stat", "readlink"):
        (binaries / tool).symlink_to(shutil.which(tool))
    monkeypatch.setenv("PATH", str(binaries))
    result = subprocess.run(
        [bash, str(script), "--with-wayland-direct"], text=True, capture_output=True
    )
    assert result.returncode == 1
    assert "Required tool 'm4' not found" in result.stderr
    assert not trace.exists()


@pytest.mark.parametrize("with_wayland", [False, True])
@pytest.mark.parametrize(
    "header", [_DEVEL_MAKEFILE.name, _CONTAINER_INTERFACE.relative_to(_DEVEL_MAKEFILE.parent)]
)
def test_missing_direct_header_fails_before_compiling(
    installer, with_wayland: bool, header: str | Path
) -> None:
    """Missing distribution interfaces stop all builds and explain host prerequisites."""
    script, trace = installer
    missing = script.parent / header
    missing.unlink()
    flags = ["--with-wayland-direct"]
    if with_wayland:
        flags.append("--with-wayland")
    result = subprocess.run(["bash", str(script), *flags], text=True, capture_output=True)
    assert result.returncode == 1
    assert f"Required SELinux development file '{missing}'" in result.stderr
    assert "sudo dnf install selinux-policy-devel" in result.stderr
    assert "sudo rpm-ostree install --allow-inactive selinux-policy-devel" in result.stderr
    assert "not in a Toolbox" in result.stderr
    assert "not to start tasks" in result.stderr
    assert "Compiling" not in result.stdout
    assert not trace.exists()


@pytest.mark.parametrize("with_wayland", [False, True])
def test_other_policies_do_not_require_direct_headers(installer, with_wayland: bool) -> None:
    """Base and filtered policy keep their existing standalone compiler requirements."""
    script, trace = installer
    (script.parent / _DEVEL_MAKEFILE.name).unlink()
    (script.parent / _CONTAINER_INTERFACE.relative_to(_DEVEL_MAKEFILE.parent)).unlink()
    flags = ["--with-wayland"] if with_wayland else []
    result = subprocess.run(["bash", str(script), *flags], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "\n-i\n" in trace.read_text()


def test_proxy_policy_limits_pipx_access_to_imports() -> None:
    """Pipx imports need reads and library maps, not writes, exec entrypoints, or container grants."""
    source = wayland_policy_source_path().read_text()
    assert "module terok_wayland 1.1;" in source
    rules = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("allow ") and "data_home_t" in line
    }
    assert rules == {
        "allow terok_wayland_t data_home_t:dir { open read };",
        "allow terok_wayland_t data_home_t:lnk_file { read getattr };",
        "allow terok_wayland_t data_home_t:file { read getattr open map execute };",
    }
    assert "allow terok_wayland_t file_type:dir { search getattr };" in source


def test_direct_policy_scopes_new_access_to_its_container_domain() -> None:
    """The opt-in domain retains the container template without global desktop grants."""
    source = wayland_direct_policy_source_path().read_text()
    rules = [line.strip() for line in source.splitlines() if not line.startswith("#")]
    assert "container_domain_template(terok_wayland_direct, container)" in rules
    assert all(
        line.startswith("allow terok_wayland_direct_t ")
        for line in rules
        if line.startswith("allow ")
    )
    assert "allow terok_wayland_direct_t terok_socket_t:unix_stream_socket connectto;" in rules
    assert "allow terok_wayland_direct_t user_tmp_t:sock_file { write getattr };" in rules
    assert "allow terok_wayland_direct_t unconfined_t:unix_stream_socket connectto;" in rules
    assert all(
        " open " not in line and " create " not in line
        for line in rules
        if line.startswith("allow ")
    )


@pytest.mark.parametrize(
    "interface",
    [
        "virt_sandbox_net_domain",
        "corenet_unconfined",
        "kernel_read_system_state",
        "logging_send_syslog_msg",
        "dev_mounton_sysfs",
        "kernel_read_messages",
    ],
)
def test_direct_policy_retains_fedora_container_baseline(interface: str) -> None:
    """The container template alone omits ordinary Fedora execution and runtime grants."""
    source = wayland_direct_policy_source_path().read_text()
    assert f"{interface}(terok_wayland_direct_t)" in source
    assert "allow terok_wayland_direct_t proc_t:filesystem remount;" in source
    assert (
        "allow terok_wayland_direct_t container_var_run_t:dir { list_dir_perms mounton };" in source
    )


@pytest.mark.parametrize(
    "boolean,grants",
    [
        (
            "virt_sandbox_use_sys_admin",
            (
                "allow terok_wayland_direct_t self:capability sys_admin;",
                "allow terok_wayland_direct_t self:cap_userns sys_admin;",
            ),
        ),
        ("virt_sandbox_use_audit", ("logging_send_audit_msgs(terok_wayland_direct_t)",)),
        (
            "container_use_xserver_devices",
            (
                "dev_getattr_xserver_misc_dev(terok_wayland_direct_t)",
                "dev_rw_xserver_misc(terok_wayland_direct_t)",
            ),
        ),
    ],
)
def test_direct_policy_keeps_stock_grants_behind_existing_booleans(
    boolean: str, grants: tuple[str, ...]
) -> None:
    """Desktop selection must not enable host capabilities or NVIDIA device access."""
    source = wayland_direct_policy_source_path().read_text()
    guarded = source.split(f"tunable_policy(`{boolean}',`", 1)[1].split("')", 1)[0]
    for grant in grants:
        assert grant in guarded
        assert source.count(grant) == 1
    assert "gen_tunable(" not in source
    assert "gen_bool(" not in source


def test_direct_policy_keeps_default_capabilities_optional() -> None:
    """The legacy capability interface follows the stock container's optional dependency."""
    source = wayland_direct_policy_source_path().read_text()
    grant = "virt_default_capabilities(terok_wayland_direct_t)"
    assert f"optional_policy(`\n    {grant}\n')" in source
    assert source.count(grant) == 1


def test_optional_policy_compiles_when_host_tools_are_available(tmp_path: Path) -> None:
    """Compile/package only; this never loads policy into the test host."""
    if not (compiler := shutil.which("checkmodule")) or not (
        packager := shutil.which("semodule_package")
    ):
        pytest.skip("optional SELinux policy compiler not installed")
    module = tmp_path / "terok_wayland.mod"
    package = tmp_path / "terok_wayland.pp"
    subprocess.run(
        [compiler, "-M", "-m", "-o", str(module), str(wayland_policy_source_path())], check=True
    )
    subprocess.run([packager, "-o", str(package), "-m", str(module)], check=True)
    assert package.stat().st_size > 0


def test_direct_policy_compiles_with_fedora_container_interfaces(tmp_path: Path) -> None:
    """Exercise the real template/compiler when Fedora development headers are installed."""
    if not _DEVEL_MAKEFILE.is_file() or not _CONTAINER_INTERFACE.is_file():
        pytest.skip("Fedora SELinux development headers not installed")
    if "container_manage_files_template" not in _CONTAINER_INTERFACE.read_text():
        pytest.skip("host container template is not the Fedora container-selinux interface")
    for tool in ("make", "m4", "checkmodule", "semodule_package"):
        if not shutil.which(tool):
            pytest.skip(f"optional SELinux build tool {tool} not installed")
    source = wayland_direct_policy_source_path()
    shutil.copyfile(source, tmp_path / source.name)
    subprocess.run(
        ["make", "-C", str(tmp_path), "-f", str(_DEVEL_MAKEFILE), "terok_wayland_direct.pp"],
        check=True,
    )
    assert (tmp_path / "terok_wayland_direct.pp").stat().st_size > 0
    expanded = (tmp_path / "tmp" / "terok_wayland_direct.tmp").read_text()
    for attribute in (
        "svirt_sandbox_domain",
        "sandbox_net_domain",
        "sandbox_caps_domain",
        "corenet_unconfined_type",
        "kernel_system_state_reader",
        "syslog_client_type",
        "mcs_constrained_type",
    ):
        assert f"typeattribute terok_wayland_direct_t {attribute};" in expanded
    assert "unconfined_domain_type" not in expanded
