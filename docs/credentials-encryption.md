# Credentials DB encryption

The credentials DB (SSH private keys, AI-provider secrets, proxy
tokens) is SQLCipher-encrypted at rest.  AES-256, mandatory — there
is no plaintext mode.

## Where the passphrase comes from

Every daemon and CLI call walks the same chain top-to-bottom and
stops at the first hit:

1. **Session-unlock file** — `$XDG_RUNTIME_DIR/terok-sandbox/vault.passphrase`,
   RAM-backed, cleared on reboot.  Written by `vault unlock`.
2. **OS keyring** — `(service=terok-sandbox, username=credentials-db)`,
   used only when `credentials.use_keyring: true` is set in `config.yml`.
3. **Config fallback** — `credentials.passphrase` in `config.yml`.
   Unsafe-on-disk; for headless hosts without a keyring.
4. **Interactive prompt** — `*`-masked, TTY only.  CLI calls; daemons
   fail loud instead.

## Day-to-day

```bash
terok-sandbox vault unlock   # asks for the passphrase, writes the
                             #   session file, restarts the daemon

terok-sandbox vault lock     # removes the session file, stops the daemon
```

`vault unlock` is normally run once per boot.

## Picking a tier at setup

`terok setup` (or `terok-sandbox setup`) asks once:

| Choice | When to pick it |
|--------|-----------------|
| `[s]` session-unlock *(default)* | desktop + laptop; one prompt per boot |
| `[k]` OS keyring                 | desktop with a working Secret Service / Keychain |
| `[c]` config file                | headless server; no keyring available |

The default is `session-unlock` — terok never touches your keyring
unless you opt in.

## Migrating a legacy plaintext DB

```bash
terok-sandbox credentials encrypt-db
```

Idempotent — re-runs on an already-encrypted DB are a no-op.

A tarred snapshot of the plaintext DB is written next to the original
as `credentials.db.plaintext-backup-<timestamp>.tar.gz` *before* the
re-key so a failed migration still has a recovery path.  The migration
prints a loud red warning afterwards — **delete the tarball with
`rm` once you have verified the new encrypted DB is good**, otherwise
your secrets stay readable on disk indefinitely.

## Doctor / sickbay

If the DB is encrypted but no tier resolves a passphrase, `terok
sickbay` reports the vault as locked with the unlock hint.  The TUI
surfaces the same state in its status line.
