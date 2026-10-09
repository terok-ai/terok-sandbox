# Audit logging

terok-sandbox writes structured JSON-lines audit logs for credential
use and shield events.  This page describes what is logged, where the
files live, and how to keep the one unbounded log rotated.

## What is logged

| Log | Scope | Written by | Path |
|-----|-------|------------|------|
| Credential-use audit | **Global** | Vault proxy | `<vault_dir>/credential_audit.jsonl` |
| Shield block audit | Per-task | NFLOG reader | `…/tasks/<project>/<task>/shield/audit.jsonl` |
| Supervisor log | Per-task | Supervisor | `…/logs/<container-id>.log` |

`<vault_dir>` defaults to
`${XDG_DATA_HOME:-~/.local/share}/terok/vault` (override with
`paths.root` or `TEROK_ROOT` in `config.yml`).

### Credential-use audit

One JSON line per credential-bearing request that crosses the vault
proxy: which scope/subject made the request, which provider, the HTTP
method and path, the outcome, and the duration.  Request and response
**bodies are never logged** — the vault is a transparent proxy and the
audit log is forensic context, not packet capture.

The file is shared across every container the vault has served.
Fields: `ts`, `scope`, `subject`, `credential_set`, `provider`,
`method`, `path`, `status`, `outcome`, `duration_ms`.

### Shield block audit

One JSON line per blocked connection attempt: destination IP, port,
protocol, resolved domain (when dnsmasq is active), and the
dossier (project/task/name).  Written by the per-container NFLOG
reader into that container's shield state directory.

## Which logs need rotation

**Per-task logs need no rotation.**  The shield block audit and the
supervisor log live inside the per-task state tree
(`…/tasks/<project>/<task>/`).  When a task ends, its state directory
is removed by `terok cleanup` — the logs are garbage-collected
naturally.  Their size is bounded by the task's lifetime.

**The credential-use audit is the only unbounded log.**  It is global,
shared across all containers, and grows for as long as the vault is in
use.  At ~200 bytes per entry and a few entries per agent request, a
busy host accumulates tens of MB per day.

## Rotating the credential audit log

terok does not run any background services, so it cannot rotate the
log itself.  If you have [`logrotate`](https://github.com/logrotate/logrotate)
installed, add a drop-in config:

```logrotate
# /etc/logrotate.d/terok-sandbox  (or ~/.config/logrotate/ for user-level)
${XDG_DATA_HOME:-$HOME/.local/share}/terok/vault/credential_audit.jsonl {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

`copytruncate` is important: the vault proxy keeps the file open in
append mode, so `copytruncate` (copy + truncate in place) keeps the
writer's file handle valid.  Without it, rotation would rename the
file and the proxy would keep writing to the orphaned inode.

For user-level rotation without root, place the config in
`~/.config/logrotate/terok-sandbox.conf` and run logrotate from a
user cron job or systemd user timer:

```bash
# ~/.config/systemd/user/terok-sandbox-logrotate.service
[Unit]
Description=Rotate terok-sandbox audit log

[Service]
Type=oneshot
ExecStart=/usr/sbin/logrotate ~/.config/logrotate/terok-sandbox.conf

# ~/.config/systemd/user/terok-sandbox-logrotate.timer
[Unit]
Description=Daily terok-sandbox audit log rotation

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now terok-sandbox-logrotate.timer
```

!!! note "Vault auditing in the supervisor model"
    The credential-use audit log is written by the vault proxy when an
    `audit_path` is configured.  In the per-container supervisor model
    the vault proxy does not currently enable auditing; the log is
    populated by standalone / CLI usage of the vault.  If you run
    terok-sandbox's vault in standalone mode and want the audit log,
    pass the path explicitly or set it up as shown above.
