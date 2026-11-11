# SELinux policies

The service-socket policy and optional filtered/direct Wayland policies are
separate modules. The installer never disables SELinux or changes the real
compositor's socket label.

```console
terok-sandbox setup selinux
terok-sandbox setup selinux --with-wayland
terok-sandbox setup selinux --with-wayland-direct
terok-sandbox setup selinux --with-wayland --with-wayland-direct
terok-sandbox setup selinux --with-wayland --show
```

The first command installs only `terok_socket`. `--with-wayland` adds
`terok_wayland` for `wl-mitm`; `--with-wayland-direct` adds
`terok_wayland_direct` for `wl-direct`. The flags are independent and can be
combined. Omitting either flag later does not remove an installed module.
`--show` prints the exact sources without installing anything. The interactive
flow displays the command and asks before running the installer with `sudo`.
The `terok` and `terok-executor` frontends expose the same commands.

Wayland still uses a Unix socket when the other services use TCP. Its optional
policy is therefore independent of the `services.mode` setting. Install the
`wl-mitm` binary separately as the desktop user to use `wl-mitm` mode; setup
does not install that binary. Direct access does not need it.

The supported modes are `none`, `wl-mitm`, and `wl-direct`. Obsolete mode names
are not aliases. Update the project configuration and recreate tasks created
with an obsolete mode; an ordinary restart cannot change the saved access grant.

## Filtered access: `--with-wayland`

The launcher explicitly selects `terok_wayland_t` for the trusted Wayland
supervisor child and its stock `wl-mitm` subprocess. It preserves the current
SELinux user, role and range. There is no
automatic transition for other invocations of the binary and no binary relabel.
The policy permits the transition under `no_new_privs` and on `nosuid` mounts.
The child enters the domain before arming parent-death signals because SELinux
clears those signals on a domain transition. The subsequent same-domain proxy
exec retains them. The trusted child can read its Python installation and task
sidecar; it sends logs through a pipe instead of obtaining home-file write access.

The domain can create the task's proxy listener, connect to the desktop
compositor, and forward the file descriptors that Wayland rendering and focused
keyboard input require. The container can connect to that proxy domain, not
directly to arbitrary `unconfined_t` sockets. Shared-memory and keymap descriptor
permissions do not grant containers permission to open host files by pathname.
The module does not add TCP/UDP, D-Bus, ptrace, or GPU-device-open permissions.
GPU passthrough remains a separate container setting.

The trusted Python child needs to read its installation before starting the
proxy. Version 1.1 permits directory listing, symlink reads and library loading
for `data_home_t`, used by pipx environments under `~/.local/share` on Fedora.
These SELinux permissions apply to the file type, not just the pipx path. They
grant no writes, relabeling or program entrypoints. The proxy's Landlock rules
remain narrower and do not expose the pipx environment to `wl-mitm`.

After upgrading from version 1.0, reload the optional policy with
`terok setup selinux --with-wayland`, then restart the affected task through
terok. A package reinstall alone does not update the loaded policy. Missing
runtime permissions can appear as `No module named terok_sandbox` in the
supervisor log even when the package imports normally in the host shell;
check the host SELinux audit log rather than disabling enforcement.

The proxy requires filesystem restrictions that narrow its executable and
writable paths further. They are not an exact-endpoint Unix-socket sandbox. SELinux also cannot
distinguish the selected compositor from another host socket with the same
process and inode labels. A compromised proxy could reach those same-label
sockets. The protocol filter and immutable configuration remain part of the
security boundary; the optional module is defense in depth, not a complete
containment guarantee.

## Direct access: `--with-wayland-direct`

This module defines `terok_wayland_direct_t`. Only `wl-direct` containers select
that process type. It combines Fedora's `container-selinux` template with the
legacy virt interfaces used by ordinary containers, retaining Podman's
per-container MCS categories. Other containers keep their existing type; this
module does not grant `container_t` permission to connect to desktop sockets.

The baseline was audited against Fedora 45's policy. It includes execution of
host-mounted container init helpers, networking, syslog, and the stock runtime
grants. Existing host booleans still control optional capabilities and audit
access; this is not an unconfined process domain. Future distribution policy
changes can require another baseline audit.

The selected container can access the mounted compositor socket and receive
rendering/keymap file descriptors. The policy also preserves its access to
terok service sockets. It does not grant pathname access to host buffer files
or enable GPU passthrough. DRM access follows the ordinary container policy;
NVIDIA access retains the existing `container_use_xserver_devices` boolean.
Installing this module neither changes that boolean nor mounts GPU devices.

Direct access has **no protocol filter**. SELinux cannot distinguish the selected
compositor from other sockets with the same process and inode labels. The exact
socket mount and container namespaces limit which endpoints are reachable; the
policy is not an endpoint-specific compositor permission.

### Host build prerequisites

The direct module uses the distribution's container-policy interfaces rather
than copying its confinement rules. Install its build prerequisites **on the
host, not in a Toolbox or task container**:

=== "Fedora"

    ```console
    sudo dnf install selinux-policy-devel policycoreutils container-selinux make m4
    ```

=== "Fedora Silverblue"

    ```console
    sudo rpm-ostree install --allow-inactive selinux-policy-devel policycoreutils container-selinux make m4
    ```

    Reboot into the new deployment before retrying the policy installer.
    `--allow-inactive` accepts packages already included in the base image.

`selinux-policy-devel` supplies `/usr/share/selinux/devel/Makefile`, and
`container-selinux` supplies its `include/services/container.if` interface.
The installer checks that both are readable before compiling any selected
module. Missing prerequisites produce installation hints; terok never installs
host packages automatically. The base and filtered policies do not require
these development files.

Development headers and build tools are needed when installing or rebuilding
the direct module, not when starting tasks. The installed container policy
remains a runtime requirement. Compilation uses a private temporary directory;
the installer does not compile unvalidated policy files found next to the
installed package.

## Failures and removal

An absent optional module leaves the administrator's existing policy in charge.
Once the module is installed, a failed domain transition stops that proxy; terok
does not retry it unconfined. Proxy or container access failures produce a
warning and leave the task running without GUI access. The filtered module does
not authorize `wl-direct` access to the real compositor. Missing direct policy
also leaves the task running with a warning, not disabled SELinux labeling.

Remove either optional module independently:

```console
sudo semodule -r terok_wayland
sudo semodule -r terok_wayland_direct
```

Package upgrades do not reload host policy. After updating the direct module,
rerun `terok setup selinux --with-wayland-direct`. Containers already using
`terok_wayland_direct_t` receive the updated rules without recreation.

Restart filtered tasks after changing their policy. Recreate direct tasks only
when changing their container process type; an ordinary restart cannot change it.
Policy changes do not retroactively change an existing process's domain. On a denial, inspect
the supervisor log and the host's AVC audit records before changing policy.
Do not disable SELinux or generate blanket `audit2allow` rules.

## Private application runtime directory

When desktop wiring is prepared, both Wayland modes set `XDG_RUNTIME_DIR` to
`/run/terok-user` inside the container. This is a private, empty tmpfs with mode
`0700`, owned by the container's initial user. It is separate from the mounted Wayland endpoint.
The tmpfs contents disappear when the container stops; Podman creates a fresh
tmpfs on the next start. The host's desktop runtime directory is never mounted.

Recreate existing containers to receive this mount and environment variable.
An ordinary restart cannot add them. No image rebuild is required.

## Host acceptance checks

Compilation and packaging do not prove an enforcing desktop works. Check on
Fedora's enforcing default desktop before relying on this policy:

- A proxy process has type `terok_wayland_t`; its container uses `container_t`.
- A direct container has type `terok_wayland_direct_t` and retains distinct MCS
  categories. Ordinary containers still cannot connect to desktop sockets.
- Normal windows, focused keyboard input, SHM rendering and GPU DMA-buf rendering
  work. A connection-only readiness check cannot validate FD transfer.
- Clipboard, capture and synthetic-input protocols remain unavailable through
  the proxy.
- Removing the module, denying the transition, or denying the container's socket
  access keeps the task alive and emits a warning instead of exposing the raw
  compositor socket.
- Stop, restart, host reboot and task removal leave no stray proxy processes.
- Direct tasks retain gate/vault access; plain restart preserves the container
  label and reports when a changed socket or label requires recreation.

Custom SELinux users, executable labels or compositor domains can require
administrator-maintained additions. Report the specific denied operation rather
than broadening the default policy speculatively.
