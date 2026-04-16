#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
#
# Install the ``terok_socket`` SELinux policy module.
#
# Compiles the ``terok_socket.te`` source that lives next to this
# script into a loadable ``.pp`` module and installs it via
# ``semodule``.  Runs as root.  Kept deliberately short and readable
# so it can be audited before invocation with ``sudo``.
#
# Invoked by ``terok setup selinux`` (which the user sees as a hint
# when ``terok setup`` detects an enforcing host with socket mode
# active and the policy not yet installed).  Can also be run directly:
#
#     sudo bash /path/to/install_policy.sh

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
te_source="${script_dir}/terok_socket.te"

if [[ ! -f "$te_source" ]]; then
    echo "Policy source not found: $te_source" >&2
    exit 1
fi

for tool in checkmodule semodule_package semodule; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool '$tool' not found." >&2
        echo "Install: dnf install selinux-policy-devel policycoreutils" >&2
        exit 1
    fi
done

workdir=$(mktemp -d -t terok-selinux-XXXXXX)
trap 'rm -rf "$workdir"' EXIT

mod="${workdir}/terok_socket.mod"
pp="${workdir}/terok_socket.pp"

checkmodule -M -m -o "$mod" "$te_source"
semodule_package -o "$pp" -m "$mod"
semodule -i "$pp"

echo "terok_socket policy installed."
