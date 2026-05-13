# Credentials DB encryption

The credentials DB (SSH private keys, AI-provider secrets, proxy
tokens) is SQLCipher-encrypted at rest.  AES-256, mandatory — there
is no plaintext mode.

## Where the passphrase comes from

Every daemon and CLI call walks the same chain top-to-bottom and
stops at the first hit:

1. **Session-unlock file** — `$XDG_RUNTIME_DIR/terok/sandbox/vault.passphrase`,
   RAM-backed, cleared on reboot.  Written by `vault unlock`.
2. **systemd-creds** — sealed credential at
   `${XDG_DATA_HOME:-~/.local/share}/terok/vault/vault.passphrase.cred`
   (`XDG_DATA_HOME` is rarely set — the `~/.local/share` fallback is
   what most hosts hit; the path matches `vault status`'s `DB:` line
   directory).  Decrypted via `systemd-creds(1)`.  Machine-bound
   (TPM2 or host key), survives reboot, no keyring needed.  Written
   by `vault seal`.  Requires systemd ≥ 257.
3. **OS keyring** — `(service=terok-sandbox, username=credentials-db)`,
   used only when `credentials.use_keyring: true` is set in `config.yml`.
4. **Config fallback** — `credentials.passphrase` in `config.yml`.
   Plaintext-on-disk; only as strong as filesystem-layer protection
   (LUKS / signed image / permissions).  `vault status` and sickbay
   permanently surface a WARNING when this tier is configured.
5. **Interactive prompt** — `*`-masked, TTY only.  CLI calls; daemons
   fail loud instead.

## Day-to-day

```bash
terok-sandbox vault unlock   # asks for the passphrase, writes the
                             #   session file, restarts the daemon

terok-sandbox vault lock     # removes the session file, stops the daemon
```

`vault unlock` is normally run once per boot.

## Picking a tier at setup

`terok setup` (or `terok-sandbox setup`) **auto-detects systemd-creds**
when the host supports it (systemd ≥ 257 with the `io.systemd.Credentials`
Varlink service) and uses it silently — that's the strongest tier
available, and asking when the answer is unambiguous just slows the
install down.  Either `host+tpm2` (TPM-equipped hosts) or `host` only,
chosen by `systemd-creds --with-key=auto`.

When systemd-creds isn't available, setup asks once:

| Choice | When to pick it |
|--------|-----------------|
| `[k]` OS keyring *(default)*      | desktop with a working Secret Service / Keychain |
| `[s]` session-unlock              | servers with no keyring; one `vault unlock` per boot |
| `[c]` config file                 | last-resort plaintext-on-disk; requires `yes` confirmation |

Either branch **auto-generates the passphrase** and echoes it once to
stderr — write it down before the install finishes, or use
`terok-sandbox vault reveal-passphrase` later (see below).

## Changing tiers (move the passphrase to a different backend)

The passphrase is one secret; the tier is just *where* it lives.
Moving it is always three steps — **retrieve, lock, reseed** — so the
ordering is the same whether you go session → keyring, keyring →
systemd-creds, or anything else.

### 1. Retrieve from the current tier

```bash
terok-sandbox vault reveal-passphrase
```

Walks the chain (same order as the daemon), prints the resolved
passphrase to **stdout**, reports the source on stderr.  Pipe-friendly,
so:

```bash
# stash in your password manager
terok-sandbox vault reveal-passphrase | pass insert -e terok/vault

# stash in a tempfile you delete after step 3
terok-sandbox vault reveal-passphrase > /tmp/vault-pw && chmod 600 /tmp/vault-pw
```

Or, if you'd rather poke each tier directly:

```bash
# session-file:
cat "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/terok/sandbox/vault.passphrase"

# OS keyring (libsecret / gnome-keyring / kwallet):
secret-tool lookup service terok-sandbox username credentials-db

# systemd-creds (sealed at rest; needs the same host that sealed it):
systemd-creds --user --name=terok-sandbox.vault-passphrase \
  decrypt "${XDG_DATA_HOME:-$HOME/.local/share}/terok/vault/vault.passphrase.cred" -

# config.yml (plaintext-on-disk):
yq '.credentials.passphrase' ~/.config/terok/config.yml
```

### 2. Lock the vault

```bash
terok-sandbox vault lock --forget
```

`--forget` clears every persistent tier in one go (session file
*plus* keyring, sealed systemd-creds, and `credentials.passphrase`).
Without it, the daemon may auto-unlock from a leftover tier on next
socket activation, defeating the swap.

### 3. Provision in the new tier

```bash
# → session-file (default; ephemeral, cleared on reboot):
echo -n "<passphrase>" | terok-sandbox vault unlock

# → systemd-creds (machine-bound, persistent):
echo -n "<passphrase>" | terok-sandbox vault unlock   # land it as session first
terok-sandbox vault seal --key=auto                   # then seal from session
terok-sandbox vault lock                              # remove the session file

# → OS keyring:
terok-sandbox setup     # chooser → [k]; reads + stores the passphrase

# → config.yml plaintext (last-resort; requires `yes` confirmation):
terok-sandbox setup     # chooser → [c] → type "yes" to accept the trust boundary
```

Run `terok-sandbox vault status` afterwards to confirm
`Passphrase: resolved via <new-tier>` — and to verify no stale
plaintext WARNING is still pointing at `config.yml`.

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
