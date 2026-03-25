# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for terok-sandbox.

Tests here exercise real servers, databases, and system interactions —
no mocks.  Environment requirements are expressed via pytest markers
so CI and the test-matrix runner can select appropriate subsets.
"""
