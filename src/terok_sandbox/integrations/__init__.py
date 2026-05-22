# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Cross-package adapters — one module per sibling wheel.

Every ``from terok_shield …`` / ``from terok_clearance …`` import in
``terok_sandbox`` must go through one of these adapters; the
``import-linter`` ``protected_modules`` contracts on the sibling
package roots enforce that.  Convention shared with terok-main, where
the same pattern lives at ``terok.lib.integrations.*``.
"""
