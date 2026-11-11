#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
#
# Install the service-socket policy and optional Wayland policies.
#
# Compiles the selected ``.te`` sources next to this script into
# loadable ``.pp`` modules and installs them via
# ``semodule``.  Runs as root.  Kept deliberately short and readable
# so it can be audited before invocation with ``sudo``.
#
# Usage:
#
#     sudo bash /path/to/install_policy.sh [--with-wayland] [--with-wayland-direct]
#
# After the policy loads, running services still hold sockets bound
# under the old context — re-run ``terok setup`` as your user so
# services rebind with ``terok_socket_t``.

set -euo pipefail

if [[ -t 1 ]]; then
    _bold=$'\033[1m' _reset=$'\033[0m' _green=$'\033[32m' _red=$'\033[31m'
else
    _bold="" _reset="" _green="" _red=""
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
devel_makefile=/usr/share/selinux/devel/Makefile
container_interface="${devel_makefile%/*}/include/services/container.if"
with_wayland=false
with_wayland_direct=false
for option in "$@"; do
    case "$option" in
        --with-wayland) with_wayland=true ;;
        --with-wayland-direct) with_wayland_direct=true ;;
        *) echo "Usage: $0 [--with-wayland] [--with-wayland-direct]" >&2; exit 2 ;;
    esac
done
modules=(terok_socket)
if "$with_wayland"; then modules+=(terok_wayland); fi
if "$with_wayland_direct"; then modules+=(terok_wayland_direct); fi

# Defence-in-depth against sudo executing attacker-tampered content: every
# file this sudo run consumes must not be swappable or rewritable by any
# user other than its owner.  Covers:
#
#  * symlink redirection — reject file links; validate both directory
#    links' containing ancestors and every target's ancestors;
#  * file rewrite — reject group/world-writable file mode bits;
#  * file replacement via directory — reject a group/world-writable parent
#    ("mv newfile oldfile" works there even on a read-only file).
#
# The files *are* legitimately user-owned (pipx, pip --user, editable
# checkouts), so we accept that but require their owner is the only writer.
_reject_unsafe() {
    local f="$1" _perm _dir _parent _dir_perm _link
    local -a _pending
    [[ "$f" == /* ]] || f="$PWD/$f"
    if [[ -L "$f" ]]; then
        echo "${_red}Refusing to run:${_reset} $f is a symlink." >&2
        echo "       A file sudo-bash'd must be a concrete regular file, not a link." >&2
        exit 1
    fi
    if [[ ! -f "$f" ]]; then
        echo "${_red}Refusing to run:${_reset} $f is not a regular file." >&2
        exit 1
    fi
    _perm=$(stat -c '%a' -- "$f") || exit 1
    if (( 8#$_perm & 8#022 )); then
        echo "${_red}Refusing to run:${_reset} $f is group- or world-writable (mode $_perm)." >&2
        echo "       A file sudo-bash'd must not be writable by any user other than its owner." >&2
        echo "       Reinstall the package into a location you control (e.g. pipx install --force)." >&2
        exit 1
    fi
    # Every ancestor, not just the immediate parent: a writable
    # grandparent lets another user rename the parent directory and
    # substitute the whole tree.  Sticky directories (/tmp) are exempt —
    # the sticky bit is exactly the rule that only the owner may rename.
    _pending=("${f%/*}")
    while (( ${#_pending[@]} )); do
        _dir="${_pending[-1]}"
        unset '_pending[-1]'
        while :; do
            while [[ "$_dir" == */ ]]; do _dir="${_dir%/}"; done
            _dir="${_dir:-/}"
            _parent="${_dir%/*}"
            _parent="${_parent:-/}"
            if [[ -L "$_dir" ]]; then
                # One hop, not realpath: intermediate links have parents
                # that disappear from the final canonical pathname.
                IFS= read -r -d '' _link < <(readlink -z -- "$_dir") || exit 1
                [[ "$_link" == /* ]] || _link="$_parent/$_link"
                _pending+=("$_link")
            else
                _dir_perm=$(stat -c '%a' -- "$_dir") || exit 1
                if (( 8#$_dir_perm & 8#022 )) && (( ! (8#$_dir_perm & 8#1000) )); then
                    echo "${_red}Refusing to run:${_reset} $_dir is group- or world-writable (mode $_dir_perm)." >&2
                    echo "       A writable ancestor lets another user replace the file via 'mv'." >&2
                    exit 1
                fi
            fi
            [[ "$_dir" == "/" ]] && break
            _dir="$_parent"
        done
    done
}
_reject_unsafe "${BASH_SOURCE[0]}"
for module in "${modules[@]}"; do
    _reject_unsafe "${script_dir}/${module}.te"
done

_prerequisite_hint() {
    echo "Install policy build prerequisites on the host, not in a Toolbox:" >&2
    echo "  Fedora: sudo dnf install selinux-policy-devel policycoreutils container-selinux make m4" >&2
    echo "  Silverblue: sudo rpm-ostree install --allow-inactive selinux-policy-devel policycoreutils container-selinux make m4" >&2
    echo "On Silverblue, reboot into the new deployment before retrying." >&2
    echo "Development headers and build tools are needed to build policy, not to start tasks." >&2
}

tools=(checkmodule semodule_package semodule)
if "$with_wayland_direct"; then tools+=(make m4); fi
for tool in "${tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "${_red}Required tool '$tool' not found.${_reset}" >&2
        _prerequisite_hint
        exit 1
    fi
done
if "$with_wayland_direct"; then
    for file in "$devel_makefile" "$container_interface"; do
        if [[ ! -f "$file" || ! -r "$file" ]]; then
            echo "${_red}Required SELinux development file '$file' is missing or unreadable.${_reset}" >&2
            _prerequisite_hint
            exit 1
        fi
    done
fi

workdir=$(mktemp -d -t terok-selinux-XXXXXX)
trap 'rm -rf "$workdir"' EXIT

packages=()
for module in "${modules[@]}"; do
    te_source="${script_dir}/${module}.te"
    mod="${workdir}/${module}.mod"
    pp="${workdir}/${module}.pp"

    echo "Compiling ${module} policy..."
    if [[ "$module" == terok_wayland_direct ]]; then
        # Refpolicy make discovers local .if/.fc inputs. Build only the
        # validated source in our private directory, never beside user files.
        cp -- "$te_source" "$workdir/"
        make -C "$workdir" -f "$devel_makefile" "${module}.pp"
    else
        checkmodule -M -m -o "$mod" "$te_source"
        echo "Packaging module..."
        semodule_package -o "$pp" -m "$mod"
    fi
    packages+=("$pp")
done

echo "Installing into system policy (rebuilds active store — may take a few seconds)..."
semodule -i "${packages[@]}"

echo "Verifying..."
installed=$(semodule -l)
for module in "${modules[@]}"; do
    if ! grep -qwF "$module" <<< "$installed"; then
        echo "${_red}ERROR:${_reset} semodule -i reported success but '${module}'" >&2
        echo "       is not in 'semodule -l'. Check the active SELinux store:" >&2
        echo "         semodule -l | grep ${module}" >&2
        echo "         getenforce" >&2
        exit 1
    fi
done

echo
echo "${_green}${modules[*]} policy installed.${_reset}"
echo
echo "Next: run ${_bold}terok setup${_reset} as your user so services rebind"
echo "their sockets with the new terok_socket_t labeling."
if "$with_wayland"; then echo "Restart wl-mitm tasks to apply the proxy domain."; fi
if "$with_wayland_direct"; then echo "Recreate existing wl-direct tasks to apply their container domain."; fi
