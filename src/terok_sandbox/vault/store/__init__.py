# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""At-rest credentials store — the SQLCipher DB and its passphrase plumbing.

The data layer behind the vault daemon.  No network, no long-lived
process, no protocol handlers; just storage primitives and the
passphrase resolution chain that unlocks the encrypted file.

Collaborators:

- [`db`][terok_sandbox.vault.store.db] — [`CredentialDB`][terok_sandbox.vault.store.db.CredentialDB]:
  the SQLite/SQLCipher store for provider secrets, SSH keys, and the
  phantom-token registry.
- [`encryption`][terok_sandbox.vault.store.encryption] — five-tier passphrase resolution chain
  (systemd-creds → keyring → kernel keyring → passphrase_command →
  interactive prompt) and the SQLCipher open / migrate primitives every
  other store module builds on.
- [`migrations`][terok_sandbox.vault.store.migrations] — schema bootstrap + forward migrations.
- [`systemd_creds`][terok_sandbox.vault.store.systemd_creds] — subprocess wrapper for ``systemd-creds(1)``,
  the machine-bound (TPM2 / host key) tier.
- [`kernel_keyring`][terok_sandbox.vault.store.kernel_keyring] — ``libkeyutils`` binding for the
  volatile kernel-keyring unlock cache tier.
"""
