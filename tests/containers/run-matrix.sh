#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
#
# Thin shim over the shared matrix engine (terok_util.matrix): this repo's
# matrix is declared in matrix.yml next to this script.  Flags are passed
# through unchanged — see `terok-matrix --help` (same surface the old
# in-repo harness had).

set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v terok-matrix >/dev/null 2>&1; then
    exec terok-matrix "$@"
fi
if command -v poetry >/dev/null 2>&1 && poetry run terok-matrix --help >/dev/null 2>&1; then
    exec poetry run terok-matrix "$@"
fi
if command -v uvx >/dev/null 2>&1; then
    exec uvx --from terok-util terok-matrix "$@"
fi
echo "terok-matrix not found: activate the repo venv (poetry install) or install uv" >&2
exit 1
