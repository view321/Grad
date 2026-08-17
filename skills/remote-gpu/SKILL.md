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
Use `--wait --timeout <seconds>` when you genuinely want to block, which from the
agent is almost never: `python -m tools.wakeup arm --run <run_id> --timeout <s>`
waits out of process and starts a new turn when the job stops, holding no shell.

## Evolve campaigns on a host

`python -m tools.evolve run --remote ssh --remote-spec <spec>` evaluates every
candidate on the host that spec names. (`--remote hf_jobs` and `--remote kaggle`
do the same thing on those backends; this section is the SSH one.) The campaign
loop — mutation, selection, the ledger — stays local. What goes to the host is
the training.

Each candidate is launched detached under `nohup` with a `grad_status.json`
marker and polled, exactly as `submit` does, and bounded by `timeout` on the host
itself. Both of those matter because a candidate is a training run: a held SSH
connection would be dropped by a NAT timeout or a sleeping laptop, and a poll
that gave up would leave the job running against the next candidate's GPU.

Each candidate gets a fresh copy of the
pipeline directory under `<workdir>/<candidate_id>`, its own `initial.py` and
`evaluate.py` written over the top, one bounded run, and the directory removed
afterwards — so candidate N cannot see what candidate N-1 left behind. A search
that can accumulate state across evaluations is one whose scores stop being
comparable, and the failure looks like a real improvement.

It refuses unless that spec has a complete, passing preflight *including the
smoke run*. That is stricter than an ordinary submission, deliberately: a
campaign is a loop with no human in it, so the environment is proven once, before
generation 0, rather than rediscovered forty times at the host's hourly rate.

Candidates do not become runs. They stay in `ledger/candidates.jsonl`; the
campaign is the ledgered unit and its expectation is the bound prediction. Cost
per candidate is measured wall clock against the host rate, not the campaign's
flat estimate — so a host priced at 0 records a campaign that spent nothing,
which is another reason to set `rate_usd_per_hour` honestly.

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
