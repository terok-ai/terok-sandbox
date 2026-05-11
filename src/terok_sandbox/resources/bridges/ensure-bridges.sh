# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — this file is deployed into containers, not used on the host.

# Idempotent socat bridge launcher for container ↔ host-side sandbox services.
#
# Manages up to four bridges:
#
#   1. SSH signer        — UNIX socket → ssh-agent-bridge.sh → TCP or host socket
#   2. Vault (HTTP leg)  — in socket mode: TCP-LISTEN → /run/terok/vault.sock
#                          (lets HTTP-only clients reach the vault via localhost)
#   3. Vault (socket leg) — in TCP mode:    /tmp/terok-vault.sock → TCP broker
#                          (lets socket-only clients — gh, claude — reach the
#                          broker, which is only exposed on TCP)
#   4. Gate server       — TCP listener → host UNIX socket (git HTTP-over-socket)
#
# Transport selection is env-var driven (set at container creation):
#
#   Socket mode: TEROK_VAULT_LOOPBACK_PORT=<port>, /run/terok/vault.sock mounted
#   TCP mode:    TEROK_TOKEN_BROKER_PORT=<port>
#
# Uses PID files (not socket existence) to detect dead bridges — stale
# socket files persist after process death and are unreliable sentinels.
#
# Designed to be *sourced* (not executed) so SSH_AUTH_SOCK propagates
# to the caller.  Typical call sites:
#   - container entrypoint (first boot)
#   - per-shell init (self-heal after restart)

_TEROK_PIDDIR=/tmp/.terok
mkdir -p "$_TEROK_PIDDIR" 2>/dev/null

# Locate ssh-agent-bridge.sh — installed alongside this script.  Resolved
# from BASH_SOURCE so the SYSTEM: invocation below works regardless of
# CWD (the script may be sourced from /, /workspace, or anywhere else).
# Falls back to a name-only invocation if BASH_SOURCE is unavailable so
# operators who put the bridges on $PATH still work.
_TEROK_BRIDGES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-./ensure-bridges.sh}")" 2>/dev/null && pwd)"
if [[ -x "${_TEROK_BRIDGES_DIR}/ssh-agent-bridge.sh" ]]; then
  _TEROK_SSH_BRIDGE="${_TEROK_BRIDGES_DIR}/ssh-agent-bridge.sh"
else
  _TEROK_SSH_BRIDGE="ssh-agent-bridge.sh"
fi

_terok_bridge_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

# ── SSH signer bridge ────────────────────────────────────────────────────
# Requires a phantom token.  Transport: TEROK_SSH_SIGNER_SOCKET (mounted
# host socket) or TEROK_SSH_SIGNER_PORT (TCP to host loopback).
if [[ -n "${TEROK_SSH_SIGNER_TOKEN:-}" ]] \
   && { [[ -n "${TEROK_SSH_SIGNER_SOCKET:-}" ]] || [[ -n "${TEROK_SSH_SIGNER_PORT:-}" ]]; } \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/ssh-agent.pid"; then
  rm -f /tmp/ssh-agent.sock
  socat "UNIX-LISTEN:/tmp/ssh-agent.sock,fork" "SYSTEM:${_TEROK_SSH_BRIDGE}" &
  echo $! > "$_TEROK_PIDDIR/ssh-agent.pid"
  export SSH_AUTH_SOCK=/tmp/ssh-agent.sock
fi

# ── Vault loopback bridge (socket mode) ──────────────────────────────────
# The host vault socket is mounted at /run/terok/vault.sock.  Socket-native
# clients (gh, claude) use it directly; everyone else reaches it via this
# TCP loopback so their "base URL" knob has something to point at.
if [[ -n "${TEROK_VAULT_LOOPBACK_PORT:-}" ]] \
   && [[ -S /run/terok/vault.sock ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/vault-loopback.pid"; then
  socat "TCP-LISTEN:${TEROK_VAULT_LOOPBACK_PORT},bind=127.0.0.1,fork,reuseaddr" \
    UNIX-CONNECT:/run/terok/vault.sock &
  echo $! > "$_TEROK_PIDDIR/vault-loopback.pid"
fi

# ── Vault socket bridge (TCP mode) ───────────────────────────────────────
# The broker is only reachable over TCP on the host.  gh and claude can
# only speak HTTP-over-UNIX, so expose a local Unix socket fronted by
# socat that pipes through to the broker.
if [[ -n "${TEROK_TOKEN_BROKER_PORT:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/vault-socket.pid"; then
  rm -f /tmp/terok-vault.sock
  socat UNIX-LISTEN:/tmp/terok-vault.sock,fork \
    TCP:host.containers.internal:"${TEROK_TOKEN_BROKER_PORT}" &
  echo $! > "$_TEROK_PIDDIR/vault-socket.pid"
fi

# ── Gate server bridge (socket mode) ─────────────────────────────────────
# In socket mode the gate HTTP server listens on a host Unix socket mounted
# into the container.  Git needs HTTP URLs, so we bridge localhost:9418 to
# the mounted socket.  CODE_REPO / CLONE_FROM point to http://localhost:9418/.
if [[ -n "${TEROK_GATE_SOCKET:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/gate.pid"; then
  socat TCP-LISTEN:9418,fork,reuseaddr \
    UNIX-CONNECT:"${TEROK_GATE_SOCKET}" &
  echo $! > "$_TEROK_PIDDIR/gate.pid"
fi
