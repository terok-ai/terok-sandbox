# Credentials DB encryption

The credentials DB (SSH private keys, AI-provider secrets, proxy
tokens) is SQLCipher-encrypted at rest.  AES-256, mandatory — there
is no plaintext mode.

## Where the passphrase comes from

Every per-container supervisor and CLI call walks the same chain
top-to-bottom and stops at the first hit:

1. **Session-unlock file** — `$XDG_RUNTIME_DIR/terok/sandbox/vault.passphrase`,
   RAM-backed, cleared on reboot.  Written by `vault unlock`.
2. **systemd-creds** — sealed credential at
   `${XDG_DATA_HOME:-~/.local/share}/terok/vault/vault.passphrase.cred`
   (`XDG_DATA_HOME` is rarely set — the `~/.local/share` fallback is
   what most hosts hit; the path matches `vault status`'s `DB:` line
   directory).  Decrypted via `systemd-creds(1)`.  Machine-bound
   (TPM2 or host key), survives reboot, no keyring needed.  Written
   by `vault passphrase seal`.  Requires systemd ≥ 257.
3. **OS keyring** — `(service=terok-sandbox, username=credentials-db)`,
   used only when `credentials.use_keyring: true` is set in `config.yml`.
4. **passphrase_command** — operator-supplied shell command set as
   `credentials.passphrase_command` in `config.yml`.  Same shape as
   `git config credential.helper`, ssh pinentry, or `BORG_PASSCOMMAND`
   — one field plugs `pass`, `bw`, `op`, `vault kv`, or any cloud
   secret-manager CLI into the resolver without per-backend code in
   the sandbox.  See [Headless setup](#headless-setup-data-center-terminals)
   below for the canonical recipe.  Fails closed when the helper is
   configured but exits non-zero / times out — silent fall-through
   to plaintext would be an unannounced security downgrade.
5. **Config fallback** — `credentials.passphrase` in `config.yml`.
   Plaintext-on-disk; only as strong as filesystem-layer protection
   (LUKS / signed image / permissions).  `vault status` and sickbay
   permanently surface a WARNING when this tier is configured.
   **Last-resort** on hosts with no systemd-creds and no usable
   helper — prefer `passphrase_command` whenever the operator has
   `pass`, `bw`, `op`, or a cloud secret-manager CLI available.
6. **Interactive prompt** — `*`-masked, TTY only.  CLI calls;
   non-interactive supervisors fail loud instead.

## Headless setup (data-center terminals)

For hosts reached over SSH where systemd-creds isn't available (older
systemd, shared cluster nodes
with no per-user TPM), point `passphrase_command` at whichever
credential helper the operator already trusts:

```yaml
credentials:
  passphrase_command: pass show terok-sandbox/vault-passphrase
```

The resolver tokenises with `shlex.split` and runs
`subprocess.run(...)` with a 30-second timeout, then strips trailing
whitespace from stdout.  Anything that prints a passphrase on stdout
works; ready-to-use recipes (🤖 guesses, not manually verified):

| Backend | `passphrase_command` value |
|---------|----------------------------|
| `pass` (gpg-agent) | `pass show terok-sandbox/vault-passphrase` |
| Bitwarden CLI | `bw get password terok-sandbox` |
| 1Password CLI | `op read op://vault/terok/passphrase` |
| HashiCorp Vault | `vault kv get -field=passphrase secret/terok` |
| AWS Secrets Manager | `aws secretsmanager get-secret-value --secret-id terok-vault --query SecretString --output text` |
| GCP Secret Manager | `gcloud secrets versions access latest --secret=terok-vault` |
| Azure Key Vault | `az keyvault secret show --vault-name terok --name vault-passphrase --query value -o tsv` |

The `pass` recipe is the canonical headless choice: gpg-agent caches
the GPG key passphrase for the session, so after one prompt each
per-container supervisor resolves it silently without re-prompting.
Provision the entry once on a
trusted workstation, then `pass git push` to your sync remote and
`pass git pull` on the data-center host — the encrypted store sits in
the same repo your dotfiles already follow.

Diagnostics from the helper land in the per-container supervisor logs
(``$XDG_STATE_HOME/terok/logs/<container-id>.log``); look for lines
like ``passphrase_command 'pass' exited 1: <stderr>``.  Each
container's supervisor walks the passphrase chain when it spawns.

## Day-to-day

```bash
terok-sandbox vault unlock   # writes the session-unlock tmpfs file
terok-sandbox vault lock     # removes the session-unlock tmpfs file
```

`vault unlock` is normally run once per boot.  The next supervisor to
start picks the freshly-resolved passphrase up automatically.

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

Either branch **auto-generates the passphrase** and prints it once
("write this down") — that's your recovery key for rebuilds and other
hosts.

## Changing tiers (move the passphrase to a different backend)

The passphrase is one secret; the tier is just *where* it lives.

### Upgrade to keyring or systemd-creds (first-class commands)

For the two most common upgrade paths — moving off the session-file
or plaintext-config tiers onto the OS keyring or a machine-bound
sealed credential — one verb does the whole swap:

```bash
# Move the passphrase from its current tier into the OS keyring.
terok-sandbox vault passphrase to-keyring

# Move it into a machine-bound systemd-creds credential.
# (Land it as session first if not already auto-resolvable, then seal.)
echo -n "<passphrase>" | terok-sandbox vault unlock
terok-sandbox vault passphrase seal --key=auto
terok-sandbox vault passphrase destroy   # clear lower-tier copies
```

`to-keyring` resolves the passphrase from whichever tier currently
holds it, validates it, writes to the keyring, flips
`credentials.use_keyring: true` in `config.yml`, drops any plaintext
fallbacks, and removes the session/sealed copies.  No retrieve-then-reseed
by hand — the next per-container supervisor to spawn resolves the
passphrase fresh from the keyring.

### Manual three-step (for anything not on the upgrade path)

For other transitions (keyring → systemd-creds, anything →
`passphrase_command`, downgrades) the swap is still **retrieve →
destroy → reseed**:

#### 1. Retrieve from the current tier


```bash
terok-sandbox vault passphrase reveal
```

Keep the value somewhere safe for the duration of the swap — a
password manager, or a `mktemp`d file you delete after step 3.

#### 2. Destroy the stored passphrase

```bash
terok-sandbox vault passphrase destroy
```

Clears every persistent tier in one go (session file *plus* keyring,
sealed systemd-creds, `credentials.passphrase`, and
`credentials.passphrase_command`).  The underlying secret stays put
in whichever store the helper points at (`pass`, 1Password, Vault,
…) — only the resolver wiring is removed.  Without this step, the
next per-container supervisor to spawn would resolve the passphrase
from a leftover tier, defeating the swap.

#### 3. Provision in the new tier

```bash
# → session-file (default; ephemeral, cleared on reboot):
echo -n "<passphrase>" | terok-sandbox vault unlock

# → systemd-creds (machine-bound, persistent):
echo -n "<passphrase>" | terok-sandbox vault unlock   # land it as session first
terok-sandbox vault passphrase seal --key=auto        # then seal from session
terok-sandbox vault lock                              # remove the session file

# → OS keyring:
terok-sandbox vault passphrase to-keyring             # one verb, no chooser

# → passphrase_command (headless; helper points at pass / bw / op / cloud CLI):
pass insert -m terok-sandbox/vault-passphrase         # or your helper's
                                                       #   "store this secret" verb
yq -yi '.credentials.passphrase_command = "pass show terok-sandbox/vault-passphrase"' \
  ~/.config/terok/config.yml

# → config.yml plaintext (last-resort; requires `yes` confirmation):
terok-sandbox setup     # chooser → [c] → type "yes" to accept the trust boundary
```

Run `terok-sandbox vault status` afterwards to confirm
`Passphrase: resolved via <new-tier>` — and to verify no stale
plaintext WARNING is still pointing at `config.yml`.

## Recovering from a lost passphrase

There is **no recovery key, no backdoor, no master key**.  The
passphrase is the only thing that unlocks the credentials DB; if every
tier loses it (you forget it, the keyring resets, the sealed
systemd-creds blob is gone with the host), the contents are
irrecoverable.

What you lose:

- Every SSH private key registered in the vault (the ones `ssh add`
  imported or generated).
- Every AI-provider credential set the agents use.
- The phantom tokens minted for in-flight container scopes.

What to do — accepting that the encrypted data is gone:

```bash
# 1. Clear any tier wiring that points at the lost passphrase.
terok-sandbox vault passphrase destroy

# 2. Delete the encrypted DB and any leftover backup tarballs.
rm "${XDG_DATA_HOME:-$HOME/.local/share}/terok/vault/credentials.db"
rm "${XDG_DATA_HOME:-$HOME/.local/share}/terok/vault/credentials.db.plaintext-backup-"*.tar.gz 2>/dev/null

# 3. Re-run setup; it mints a fresh passphrase and creates a clean DB.
terok-sandbox setup
```

After step 3 you have an empty vault — re-import your SSH keys, re-add
provider credentials.

**Back the passphrase up *before* you lose it.**

## Doctor / sickbay

If the DB is encrypted but no tier resolves a passphrase, `terok
sickbay` reports the vault as locked with the unlock hint.  The TUI
surfaces the same state in its status line.
