#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
#
# Install the terok-shield AppArmor addendum for the per-container dnsmasq.
#
# Adds an owner-scoped rule block to the host's dnsmasq AppArmor profile
# (via its ``local/`` include) permitting the per-task shield state tree,
# then reloads the profile.  No compilation — just ``apparmor_parser -r``.
# Kept short and readable so it can be audited before invocation with sudo.
#
# Usage:
#
#     sudo bash /path/to/install_profile.sh <STATE_ROOT>
#
# <STATE_ROOT> is the sandbox-live dir (e.g. ~/.local/share/terok/sandbox-live).
# It must be passed: under sudo this script cannot resolve the operator's
# home, and AppArmor mediates by pathname so the rules must name the root.

set -euo pipefail

if [[ -t 1 ]]; then
    _bold=$'\033[1m' _reset=$'\033[0m' _green=$'\033[32m' _red=$'\033[31m'
else
    _bold="" _reset="" _green="" _red=""
fi

state_root="${1:-}"
if [[ -z "$state_root" ]]; then
    echo "${_red}Usage:${_reset} sudo bash $0 <STATE_ROOT>" >&2
    echo "       <STATE_ROOT> = your sandbox-live dir, e.g. ~/.local/share/terok/sandbox-live" >&2
    exit 2
fi
state_root="${state_root%/}"

# Defence-in-depth against sudo executing attacker-tampered content: a file
# sudo-bash'd must not be swappable or rewritable by any user other than its
# owner.  Reject symlinks, non-regular files, group/world-writable files,
# and group/world-writable parent dirs (a writable parent allows 'mv').
_self="${BASH_SOURCE[0]}"
if [[ -L "$_self" ]]; then
    echo "${_red}Refusing to run:${_reset} $_self is a symlink." >&2
    exit 1
fi
if [[ ! -f "$_self" ]]; then
    echo "${_red}Refusing to run:${_reset} $_self is not a regular file." >&2
    exit 1
fi
if (( 8#$(stat -c '%a' "$_self") & 8#022 )); then
    echo "${_red}Refusing to run:${_reset} $_self is group- or world-writable." >&2
    exit 1
fi
if (( 8#$(stat -c '%a' "$(dirname "$_self")") & 8#022 )); then
    echo "${_red}Refusing to run:${_reset} parent of $_self is group- or world-writable." >&2
    exit 1
fi

if ! command -v apparmor_parser >/dev/null 2>&1; then
    echo "${_red}apparmor_parser not found.${_reset} Install the 'apparmor' package." >&2
    exit 1
fi

# Locate the stock dnsmasq profile (profile-set dependent) and its local include.
profile=""
for p in /etc/apparmor.d/usr.sbin.dnsmasq /etc/apparmor.d/dnsmasq; do
    if [[ -f "$p" ]]; then profile="$p"; break; fi
done
if [[ -z "$profile" ]]; then
    echo "${_red}No dnsmasq AppArmor profile found${_reset} in /etc/apparmor.d." >&2
    echo "       dnsmasq is not AppArmor-confined on this host; nothing to install." >&2
    exit 1
fi
local_include="/etc/apparmor.d/local/$(basename "$profile")"

# Idempotent: strip any prior managed block, then append the freshly
# rendered one.  Owner-scoping limits the grant to files dnsmasq owns; the
# glob covers every per-task shield dir under STATE_ROOT.
mkdir -p "$(dirname "$local_include")"
if [[ -f "$local_include" ]]; then
    sed -i '/# >>> terok-shield apparmor/,/# <<< terok-shield apparmor/d' "$local_include"
fi
cat >> "$local_include" <<EOF
# >>> terok-shield apparmor (managed; do not edit between markers) >>>
owner ${state_root}/tasks/*/*/shield/dnsmasq.conf r,
owner ${state_root}/tasks/*/*/shield/dnsmasq.pid rwk,
owner ${state_root}/tasks/*/*/shield/dnsmasq.log rwk,
/usr/share/iproute2/* r,
# <<< terok-shield apparmor (managed) <<<
EOF

echo "Reloading ${profile} ..."
apparmor_parser -r -W "$profile"

echo
echo "${_green}terok-shield AppArmor addendum installed.${_reset}"
echo "Profile: ${_bold}${profile}${_reset}  (rules added to ${local_include})"
echo "Re-run your task — the dnsmasq DNS tier should now be used."
