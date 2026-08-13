---
name: remote-gpu
description: SSH host conventions, the host inventory, and how a job is staged, launched, watched, and collected by gpu.py. Load before the first remote run on a new host.
---

# Remote GPU hosts

`gpu.py` is not a wrapper around `ssh`. It is a small set of operations over a
fixed inventory, and that difference is the security model: an allowlist over
our own operations is enforceable in a way that an allowlist over a shell is not.

## The inventory

Hosts live in `config/grad.toml` and an unknown name is a configuration error,
never an ad-hoc connection.

```toml
[hosts.gpu-box]
hostname          = "10.0.0.7"
user              = "research"
gpus              = 2
rate_usd_per_hour = 0.0        # 0 = free to use; still ledgered, just at zero
workdir           = "~/grad"
key_credential    = "gpu_box_key"   # a Credential Manager entry, not a path
notes             = "2x4090, shared with the lab"
```

`rate_usd_per_hour` is what `collect` prices wall clock against — SSH hosts have
no billing API, so this table *is* the accounting. Set it honestly; a host
priced at 0 that actually costs money makes the monthly ceiling a fiction.

## Credentials

Keys live in Windows Credential Manager, never in the workspace and never in the
agent's environment:

```bash
python -m tools.jobs credential set gpu_box_key   # prompts, does not echo
```

`gpu.py` materialises the key to a mode-600 file in the OS temp directory for
the duration of one call and deletes it afterwards. That is weaker than never
materialising it, and it is why an SSH agent or a `~/.ssh/config` host entry is
preferable where you have one — leave `key_credential` unset and no key is ever
written to disk by us.

## What a run does

1. `submit` runs the four gates, writes the in-flight run record, `scp`s the
   pipeline directory to `<workdir>/<run_id>`, and launches the command under
   `nohup`, writing a `grad_status.json` marker when it finishes.
2. `status` reads that marker. It never scrapes logs.
3. `collect` pulls `stdout.log`, `stderr.log`, the metrics file, and anything in
   `config.artifact_paths`; prices wall clock against the host rate; writes the
   run record with a mechanically-computed `deviations` array; and removes the
   remote directory unless `--keep-remote`.

`collect` is non-blocking by default and exits 10 while the job is running — a
two-hour poll inside the agent's only shell is a tool timeout waiting to happen.
Use `--wait --timeout <seconds>` when you genuinely want to block.

## Conventions the pipeline must follow

- Write a machine-readable metrics artifact. A JSON object of
  `quantity -> value`, or JSONL of `{"quantity": ..., "value": ...}` records.
  This is a contract, not a nicety: it removes all log-scraping, and `collect`
  fails loudly without it.
- Accept `--steps` and `--smoke`. The smoke path passes both, and the caps are
  enforced by the submitter regardless of what the pipeline does with them.
- Write artifacts under the run directory, not to absolute paths.

## Multi-GPU

Launch with whatever the pipeline expects (`torchrun`, `accelerate`) by setting
`target.command` in the spec. The smoke run uses the same command, which is the
point — distributed initialisation is single-process locally and therefore
invisible to the local dry run.
