#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025-2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — this file is deployed into containers, not used on the host.

# Relay between an SSH agent client (via socat SYSTEM: stdin/stdout) and
# the host-side SSH signer, injecting the phantom token handshake
# as the first bytes on the connection.
#
# Called by socat:
#   socat UNIX-LISTEN:...,fork "SYSTEM:ssh-agent-bridge.sh"
#
# stdin/stdout = the SSH client's Unix socket side (provided by socat)
#
# Transport is selected by env vars (mutually exclusive):
#   TEROK_SSH_SIGNER_SOCKET - Unix socket path (socket mode, mounted from host)
#   TEROK_SSH_SIGNER_PORT   - TCP port on host.containers.internal (TCP mode)
#
# Always required:
#   TEROK_SSH_SIGNER_TOKEN  - phantom token for the handshake

set -euo pipefail

: "${TEROK_SSH_SIGNER_TOKEN:?missing}"

# Resolve upstream target: socket takes precedence over TCP.
#
# retry=/interval= make socat hold each agent request and re-attempt the
# connect until the supervisor has bound the signer, rather than failing
# instantly when the container's first task-init clone runs before the
# signer is up (the same readiness race the gate bridge handles).  The
# supervisor binds the signer once its vault DB is open, so this usually
# connects on the first try; the retry only matters during startup.
_RETRY="retry=30,interval=1"
if [[ -n "${TEROK_SSH_SIGNER_SOCKET:-}" ]]; then
  TARGET="UNIX-CONNECT:${TEROK_SSH_SIGNER_SOCKET},${_RETRY}"
elif [[ -n "${TEROK_SSH_SIGNER_PORT:-}" ]]; then
  [[ "${TEROK_SSH_SIGNER_PORT}" =~ ^[0-9]+$ ]] || {
    echo "TEROK_SSH_SIGNER_PORT must be numeric" >&2
    exit 2
  }
  TARGET="TCP:host.containers.internal:${TEROK_SSH_SIGNER_PORT},${_RETRY}"
else
  echo "One of TEROK_SSH_SIGNER_SOCKET or TEROK_SSH_SIGNER_PORT is required" >&2
  exit 2
fi

# Compute 4-byte big-endian length header dynamically — the server-side
# _read_handshake() accepts any token length 1–1024 and does a DB lookup.
TOKEN_LEN=${#TEROK_SSH_SIGNER_TOKEN}

# Send the token handshake, then relay SSH agent traffic bidirectionally.
{
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 24 & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 16 & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 8  & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN       & 0xFF)))"
  printf '%s' "${TEROK_SSH_SIGNER_TOKEN}"
  cat
} | socat - "${TARGET}"
