# Credentials DB encryption

The credentials DB (SSH private keys, AI-provider secrets, proxy
tokens) is SQLCipher-encrypted at rest.  AES-256, mandatory — there
is no plaintext mode.

## Where the passphrase comes from

Every per-container supervisor and CLI call walks the same chain
top-to-bottom and stops at the first hit:

1. **systemd-creds** — sealed credential at
   `${XDG_DATA_HOME:-~/.local/share}/terok/vault/vault.passphrase.cred`
   (the same directory as the credentials DB).  Decrypted via
   `systemd-creds(1)`.  Machine-bound
   (TPM2 or host key), survives reboot, no desktop keyring needed.  Written
   by `vault passphrase seal`.  Requires systemd ≥ 257.
2. **Desktop keyring** — `(service=terok-sandbox, username=credentials-db)`.
   On by default (an empty desktop keyring simply doesn't resolve); set
   `credentials.use_desktop_keyring: false` in `config.yml` to keep the chain
   away from Secret Service entirely. GNOME Keyring is one implementation;
   "login" can name a desktop-keyring collection, not a separate storage tier.
3. **Session cache** — the **kernel keyring** (Linux's per-UID kernel user
   keyring, `@u`), with a tmpfs session-file fallback when the supervisor
   cannot access it. Written by `vault unlock`. It never survives reboot
   and can disappear earlier. Kernel-keyring lifetime depends on processes
   and references: logout alone does not guarantee clearing it.
4. **passphrase_command** — operator-supplied shell command set as
   `credentials.passphrase_command` in `config.yml`.  Same shape as
   `git config credential.helper`, ssh pinentry, or `BORG_PASSCOMMAND`
   — one field plugs a plain secret file (`cat /path/to/file`),
   `pass`, `bw`, `op`, `vault kv`, or any cloud secret-manager CLI
   into the resolver without per-backend code in the sandbox.  See
   [Headless setup](#headless-setup-data-center-terminals)
   below for the canonical recipe.  Fails closed when the helper is
   configured but exits non-zero / times out — silent fall-through
   to a weaker tier would be an unannounced security downgrade.
5. **Interactive prompt** — `*`-masked, TTY only.  CLI calls;
   non-interactive supervisors fail loud instead.

The desktop keyring supplies the passphrase directly to the process opening
SQLCipher; it does not need an additional kernel-keyring copy. Manual
unlock uses the session cache to avoid retyping the passphrase in later
processes. Both paths still bring secret material into process memory.
The CLI/config tier IDs are `desktop-keyring` for the desktop keyring and
`session-cache` for the temporary cache. The session-cache tier is not
always backed by the kernel keyring: `vault status` identifies the actual
backend as the kernel keyring or a tmpfs session file.

!!! warning "Storage names have no compatibility aliases"
    Rename the old desktop-keyring setting `credentials.use_keyring` to
    `credentials.use_desktop_keyring` in existing configuration. The old
    setting is rejected. Use `to-desktop-keyring` for the transfer command,
    and `desktop-keyring` or `session-cache` for `--passphrase-tier`.
    Recreate existing task containers to regenerate their sidecars with
    the renamed `credentials_use_desktop_keyring` policy field.
    Stored desktop-keyring entries and kernel-keyring cache identities
    are unchanged; this rename does not require moving the passphrase.

!!! note "The plaintext `credentials.passphrase` tier was removed"
    Configs that still set it are rejected with migration directions:
    move the value into its own file (mode 600) and point
    `passphrase_command: cat /path/to/that/file` at it.  Same trust
    boundary (filesystem-level protection), one tier instead of two.

## Where the supervisor runs

One host fact places the per-container supervisor, and the OCI hook reads it
the same way the launcher does: a per-user systemd manager that answers.
With one, the supervisor is a transient unit of that manager
(`terok-supervisor-<12-character container id>.service`), in the operator's own namespaces.
Without one, it is a daemon inside the container runtime's user namespace.
The volatile cache tier follows the placement.
A user-unit supervisor reads the operator's kernel keyring, so that is the
backing.
A namespace daemon sees an empty kernel keyring there, so the backing is the session
file under `$XDG_RUNTIME_DIR`, a path being a path in any namespace.
Without a runtime directory the cache tier refuses, and the supervisor's
children need `credentials.passphrase_command` instead.
`vault status` names the backing and the reason; `sickbay` names the placement.
The user unit carries the hardening a user service can take without a user
namespace: no new privileges, no realtime, no personality change, native
system calls, a private umask.
Filesystem sandboxing is deliberately absent: a user service gets it only
through `PrivateUsers`, a fresh user namespace where the kernel keyring is empty
again.
Units of a user manager stop at the last logout unless `loginctl
enable-linger` is set, the rule rootless podman documents for the containers
themselves.

## Runtime service boundaries

The per-container supervisor runs vault, signer, gate, and clearance in
separate processes.  Outside debugger mode, Landlock also limits each
process to its own filesystem lane: service sockets have distinct writable
directories, vault and signer can open the credentials DB, and a gate can
mutate only its scoped bare mirror.  The verdict broker is deliberately
exempt because its job requires Podman runtime state.

This is a filesystem-path boundary, not a same-UID kernel-keyring boundary.
Vault and signer both resolve the credentials passphrase through the
configured desktop-keyring policy by design. Operators who do not want that shared
desktop-keyring trust can disable `credentials.use_desktop_keyring` and choose another tier.
The launch sidecar snapshots that non-secret policy.  Each secret-holder
resolves it after process hardening but before installing Landlock, so an
operator-selected `passphrase_command` may use its normal files while the
long-lived service cannot.
On kernels where the complete Landlock policy is unavailable, startup logs
whether confinement was partial or not applied.

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
`subprocess.run(...)` with a 30-second timeout, then strips the
trailing newline from stdout (inner whitespace reaches SQLCipher
verbatim).  Anything that prints a passphrase on stdout
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

Helper failure summaries land in the per-container supervisor logs
(``<state_root>/logs/<container-id>.log``, state root defaulting to
``~/.local/share/terok/sandbox``); look for lines like
``passphrase_command exited 1``. Command text, stdout, stderr, and raw
exception messages are never logged: any of them could contain the
passphrase. Each container's supervisor walks the passphrase chain when
it spawns.

## Day-to-day

```bash
terok-sandbox vault unlock   # validates and caches in kernel keyring or tmpfs session file
terok-sandbox vault lock     # deletes saved passphrases; running services stay open
```

`vault unlock` is needed again after the temporary cache disappears.  The typed value is
**validated against the existing credentials DB first** — a wrong entry
exits with an error and writes nothing, so a typo can't silently park a
useless key in the session cache.  (With no DB yet there is
nothing to validate; the value becomes the encryption key on first
use.)  The next supervisor to start picks the freshly-resolved
passphrase up automatically.

`unlock` skips creating a redundant cache when a durable tier already
supplies the passphrase (systemd-creds, desktop keyring, or the configured
helper). Pass `--force` only when deliberately refreshing the temporary
cache. The cache sits below systemd-creds and the desktop keyring, so it
does not override either of those sources.

`vault lock` removes both temporary backings, the saved desktop-keyring
entry, and the sealed systemd-creds credential. It disconnects
`passphrase_command` without deleting the external secret that helper
reads. **This is destructive passphrase removal, not desktop-keyring
locking. Already-running services remain open; their access is not
revoked.** Keep a recovery copy before using it.

## Picking a tier at setup

`terok setup` (or `terok-sandbox setup`) **auto-detects systemd-creds**
when the host supports it (systemd ≥ 257 with the `io.systemd.Credentials`
Varlink service) and uses it silently — that's the strongest tier
available, and asking when the answer is unambiguous just slows the
install down.  Either `host+tpm2` (TPM-equipped hosts) or `host` only,
chosen by `systemd-creds --with-key=auto`.

When systemd-creds isn't available, setup asks once (non-interactive
runs must pass `--passphrase-tier` explicitly — there is no silent
fallback):

| Choice | When to pick it |
|--------|-----------------|
| `[k]` Desktop keyring *(default)* | desktop with a working Secret Service / Keychain |
| `[n]` Session cache (kernel keyring or tmpfs session file) | no persistent store; re-enter after reboot or cache loss |

(Headless hosts that want a file-based store skip the chooser and set
`credentials.passphrase_command: cat /path/to/secret-file` instead —
see [Headless setup](#headless-setup-data-center-terminals).)

Either branch **auto-generates the passphrase** and prints it once
("write this down") — that's your recovery key for rebuilds and other
hosts.

## Changing the passphrase

```bash
terok-sandbox vault passphrase change
```

One verb does the whole rotation: it resolves the current passphrase
from the chain (prompting only when the vault is locked — retyping a
value the same shell can print with `vault passphrase reveal` would be
theatre), asks for the new one (typed-and-confirmed, or Enter to
generate), **re-encrypts the credentials DB** under the new key, and
rewrites every tier that stored the old one.  The recovery
acknowledgement is dropped and re-run — the copy you saved before is
now the wrong passphrase.

Ground rules the verb enforces:

- **Failures before the re-encryption change nothing.**  A wrong
  current passphrase or a busy DB aborts with every tier and the DB
  exactly as they were.
- **Failures after the re-encryption can't lose the key.**  The new
  value is escrowed to a RAM-backed, owner-only pending file *before*
  the DB is rekeyed and deleted once at least one tier holds it — so
  even a crash mid-change leaves the new key recoverable on the host.
  A tier that can't take the new value (desktop keyring denied, systemd-creds
  host regressed) is **purged and reported** rather than left holding
  the old passphrase, and the verb exits non-zero so the failure can't
  scroll past.
- **Running tasks hold the vault open.**  Re-encryption needs
  exclusive access; with a live task the verb refuses with a
  `database is locked` hint (`fuser -v` the DB, stop the task, re-run).
  Tasks that resolved the old passphrase keep using it until
  restarted; new tasks pick up the new one automatically — `terok
  sickbay` flags the stale ones.
- **`passphrase_command` blocks the change.**  The helper's secret
  lives in a store you own (`pass`, 1Password, a cloud vault) that
  terok cannot write to — update the secret there first, or remove
  the `passphrase_command` wiring, then re-run.

## Changing tiers (move the passphrase to a different backend)

The passphrase is one secret; the tier is just *where* it lives.

### Upgrade to desktop keyring or systemd-creds (first-class commands)

For the two most common upgrade paths — moving off the temporary
cache onto the desktop keyring or a machine-bound sealed credential —
one verb does the whole swap:

```bash
# Move the passphrase from its current tier into the desktop keyring.
terok-sandbox vault passphrase to-desktop-keyring

# Move it into a machine-bound systemd-creds credential.
# (Land it as session first if not already auto-resolvable, then seal.
#  `seal` drops the now-redundant session copy for you.)
echo -n "<passphrase>" | terok-sandbox vault unlock
terok-sandbox vault passphrase seal --key=auto
```

`to-desktop-keyring` resolves the passphrase from whichever tier currently
holds it, validates it, writes to the desktop keyring, and verifies that
the saved value can be read back without prompting before removing the
source. It enables `credentials.use_desktop_keyring` in `config.yml`, disconnects
`passphrase_command`, and removes the temporary/sealed copies. If the
write or readback fails, the source is preserved. The running TUI and the
next supervisor use the updated configuration.

Both upgrade verbs refuse to enable a machine-bound auto-unlock tier
until the recovery key is marked as saved — run
`terok-sandbox vault passphrase reveal` (which offers to mark it) or
`vault passphrase acknowledge` first.

### Manual three-step (for anything not on the upgrade path)

For other transitions (anything → `passphrase_command`, downgrades) the swap is still **retrieve →
lock → reseed**:

#### 1. Retrieve from the current tier


```bash
terok-sandbox vault passphrase reveal
```

Keep the value somewhere safe for the duration of the swap — a
password manager, or a `mktemp`d file you delete after step 3.
`reveal` offers to mark the recovery key as saved; accept it so the
`lock` in step 2 doesn't stop to ask.

#### 2. Lock — clear every stored copy

```bash
terok-sandbox vault lock
```

Clears every tier in one go (kernel keyring and tmpfs session cache,
desktop keyring, sealed systemd-creds, and the `credentials.passphrase_command`
wiring) and drops the recovery marker. The underlying secret stays put in whichever store the helper points at
(`pass`, 1Password, Vault, …) — only the resolver wiring is removed.
Without this step, the next per-container supervisor to spawn would
resolve the passphrase from a leftover tier, defeating the swap.  An
unconfirmed vault asks you to type `SAVED` first (or pass `--force`);
you just retrieved the value in step 1, so you have it.

#### 3. Provision in the new tier

The machine-bound targets (systemd-creds, desktop keyring) refuse until the
recovery key is re-acknowledged — the `lock` in step 2 dropped the
marker — so run `terok-sandbox vault passphrase acknowledge` first
for those two paths.

```bash
# → temporary cache (kernel keyring or tmpfs; lost at reboot or earlier):
echo -n "<passphrase>" | terok-sandbox vault unlock

# → systemd-creds (machine-bound, persistent):
echo -n "<passphrase>" | terok-sandbox vault unlock   # land it as session first
terok-sandbox vault passphrase acknowledge            # re-confirm the recovery key
terok-sandbox vault passphrase seal --key=auto        # seal; drops the session copy

# → desktop keyring:
terok-sandbox vault passphrase acknowledge            # re-confirm the recovery key
terok-sandbox vault passphrase to-desktop-keyring     # one verb, no chooser

# → passphrase_command (headless; helper points at a file, pass / bw / op / cloud CLI):
pass insert -m terok-sandbox/vault-passphrase         # or your helper's
                                                       #   "store this secret" verb
yq -yi '.credentials.passphrase_command = "pass show terok-sandbox/vault-passphrase"' \
  ~/.config/terok/config.yml
```

Run `terok-sandbox vault status` afterwards to confirm the header
reads `Vault: unlocked — passphrase via <new-tier>`.

## Recovering from a lost passphrase

There is **no recovery key, no backdoor, no master key**.  The
passphrase is the only thing that unlocks the credentials DB; if every
tier loses it (you forget it, the desktop keyring resets, the sealed
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
#    --force: the passphrase is gone, so you can't confirm a saved copy.
terok-sandbox vault lock --force

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
