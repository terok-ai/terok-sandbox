# Changelog

Per-release notes live on the [GitHub Releases page][rel], which is
also the "Changelog" link in PyPI metadata.

[rel]: https://github.com/terok-ai/terok-sandbox/releases

## v0.0.123 — SELinux Rules Hint

## What's Changed
* feat(setup): re-surface SELinux install command + TCP alternative at end of setup in https://github.com/terok-ai/terok-sandbox/pull/298

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.122...v0.0.123

## v0.0.122 — The State of the Shields

## What's Changed
* feat!: rename shield.block adapter to shield.quarantine in https://github.com/terok-ai/terok-sandbox/pull/299
* chore(deps): drop stale clearance branch-pin comment by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/297


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.121...v0.0.122

## v0.0.121 — The Tree of Command

## What's Changed
* feat(cli): unified CommandTree + structural nesting (no more flat tuples) in https://github.com/terok-ai/terok-sandbox/pull/295
* feat(vault): consolidate passphrase management under `vault passphrase` namespace by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/294
* test(matrix): pin exact podman versions, warn on drift; refresh dev deps by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/296


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.120...v0.0.121

## v0.0.120 — Sandbox API restructure

## What's Changed
* refactor: drop the vault/credentials back-compat shims in https://github.com/terok-ai/terok-sandbox/pull/293
* feat(vault): pluggable passphrase_command chain tier (closes #283) by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/290
* refactor: split commands.py into per-subsystem package; tighten vocabulary by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/291
* refactor: unify vault and credentials under vault/{store,ssh,daemon} by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/292


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.119...v0.0.120

## v0.0.119 — Passphrase Handling

## What's Changed
* feat(vault): permanent plaintext-passphrase visibility (#282) by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/287


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.118...v0.0.119

## v0.0.118 — Trusted Platform Module

## What's Changed
* ci: split SonarQube into its own workflow; add fork-PR sonar; simplify docs wait by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/274
* ci(sonar): only allow workflow_dispatch from master by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/275
* feat(vault): enrich VaultStatus with ssh_keys_stored, passphrase_source, locked by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/278
* feat(vault): systemd-creds tier between session-unlock and OS keyring by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/281
* test(matrix): align runner with terok-shield; +ubuntu26.04 +fedora44 by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/280
* feat(vault): rename SSH key comment via CredentialDB.set_ssh_key_comment by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/285
* test(credentials): integration matrix asserting against new VaultStatus fields by @sliwowitz in https://github.com/terok-ai/terok-sandbox/pull/284


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.117...v0.0.118

## v0.0.117 — Vault Passphrase

Credentials Vault DB encryption

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.116...v0.0.117

## v0.0.116 — Checkpoint Charlie

## What's Changed
* seed CHANGELOG.md for the first PyPI release in https://github.com/terok-ai/terok-sandbox/pull/262

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.115...v0.0.116

## v0.1.0 — first PyPI release

`terok-sandbox` joined PyPI in v0.1.0. Versions before that were GitHub Release
wheels only.
