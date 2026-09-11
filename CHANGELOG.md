# Changelog
## v0.5.0 — Past Prologue

## What's Changed
* fix: stoppable task containers — --init pid1 + state-watching stop() in https://github.com/terok-ai/terok-sandbox/pull/454
* feat(vault): programmatic passphrase-tier provisioning API for TUI frontends in https://github.com/terok-ai/terok-sandbox/pull/456
* feat(gpu): per-device grants + vendor-scoped raw DRM mounts in https://github.com/terok-ai/terok-sandbox/pull/463
* fix(gpu): by-path binds must use --mount — PCI addresses break -v parsing in https://github.com/terok-ai/terok-sandbox/pull/464
* feat(gate): agent-op backups + push marker via sandbox-owned hooks in https://github.com/terok-ai/terok-sandbox/pull/483
* feat(vault): replace session-file tier with a kernel-keyring cache in https://github.com/terok-ai/terok-sandbox/pull/489
* refactor!: rename the shield kill-switch to 'disabled' and the DOWN flag to 'disengaged' in https://github.com/terok-ai/terok-sandbox/pull/521
* feat: user-friendly hints for AppArmor and SELinux in https://github.com/terok-ai/terok-sandbox/pull/537
* feat: forward credential metadata as OAuth headers in https://github.com/terok-ai/terok-sandbox/pull/548


**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.4.1...v0.5.0

## v0.4.1 — You Exist Here

* Resilient clone-cache refresh, https://github.com/terok-ai/terok-sandbox/pull/430
* Fast cold-start reconnect for gate + retry on vault bridges, https://github.com/terok-ai/terok-sandbox/pull/426
* Resilient gate restart after upgrade https://github.com/terok-ai/terok-sandbox/pull/428

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.4.0...v0.4.1

## v0.4.0 — The Celestial Temple

* Better UI for vault passphrase handling, https://github.com/terok-ai/terok-sandbox/pull/408, https://github.com/terok-ai/terok-sandbox/pull/399, https://github.com/terok-ai/terok-sandbox/pull/402, https://github.com/terok-ai/terok-sandbox/pull/400, https://github.com/terok-ai/terok-sandbox/pull/398
* Container lifecycle verbs, https://github.com/terok-ai/terok-sandbox/pull/411
* Operator hints https://github.com/terok-ai/terok-sandbox/pull/417, https://github.com/terok-ai/terok-sandbox/pull/419

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.3.1...v0.4.0

## v0.3.2 — We are constantly searching

## What's Changed
* Vault diagnostics improved https://github.com/terok-ai/terok-sandbox/pull/408, https://github.com/terok-ai/terok-sandbox/pull/399, https://github.com/terok-ai/terok-sandbox/pull/400
* Remote API fix: preserve URL-encoded path segments when forwarding, https://github.com/terok-ai/terok-sandbox/pull/402

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.3.1...v0.3.2

## v0.3.1 — Start Again

hotfix: rebuild the per-container runtime dir in Sandbox.start, https://github.com/terok-ai/terok-sandbox/pull/396

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.3.0...v0.3.1

## v0.3.0 — Locks and Hooks

Hotfix for vault passphrase error modes [#390](https://github.com/terok-ai/terok-sandbox/pull/390) and supervisor restart [#389](https://github.com/terok-ai/terok-sandbox/pull/389)

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.2.0...v0.3.0

## v0.2.0 — Emissary, Part II

## What's Changed
* Combining LLM providers × agents, https://github.com/terok-ai/terok-sandbox/pull/374
* podman container event stream, https://github.com/terok-ai/terok-sandbox/pull/375
* SSH key M:N routing to projects, https://github.com/terok-ai/terok-sandbox/pull/385
* Vault debugging and safer locking, https://github.com/terok-ai/terok-sandbox/pull/386

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.1.0...v0.2.0

## v0.1.0 — The Emissary

**First public PyPi release**

## What's Changed

* resolve gate binary via sys.executable -m, like vault, https://github.com/terok-ai/terok-sandbox/pull/309
* accept %host-style infrastructure scopes, https://github.com/terok-ai/terok-sandbox/pull/320
* KrunRuntime — KVM microVM isolation backend (Phase 3), https://github.com/terok-ai/terok-sandbox/pull/324
* Install PYTHONPATH to SystemD service files by @franzpoeschel in https://github.com/terok-ai/terok-sandbox/pull/338
* add check_gpu_available(), https://github.com/terok-ai/terok-sandbox/pull/362
* AppArmor profile installer for the dnsmasq DNS tier, https://github.com/terok-ai/terok-sandbox/pull/367
* per-container supervisor; retire host vault + gate daemons, https://github.com/terok-ai/terok-sandbox/pull/366

## New Contributors
* @franzpoeschel made their first contribution in https://github.com/terok-ai/terok-sandbox/pull/338

**Full Changelog**: https://github.com/terok-ai/terok-sandbox/compare/v0.0.123...v0.1.0

