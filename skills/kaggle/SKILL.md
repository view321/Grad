---
name: kaggle
description: Kaggle's free GPU/TPU backend — the weekly accelerator allowance, how a pipeline is packed into one notebook, and what a spec must declare to be submitted there. Load before the first Kaggle run, or after an exit 13.
---

# Kaggle kernels

`kaggle.py` is the third submitter, and the first whose scarce resource is not
money. The four gates of §6 all run and the spend gate passes every time,
because a Kaggle kernel costs $0.00 and that is under every ceiling there is.
**Do not read that as headroom.** What Kaggle rations is accelerator hours, and
that is enforced separately, with its own exit code.

## The allowance

| pool | weekly | one session |
|---|---|---|
| GPU (P100, T4, T4 high-mem, A100, L4, H100, RTX Pro 6000) | ~30 h | 12 h |
| TPU (v3-8, v5e-8, v6e-8) | ~20 h | 9 h |

Both ceilings refuse with **exit 13**, and they fail differently:

- **weekly** — the pool is shared by every run in a rolling 7-day window.
  Counted as actual hours for collected runs and *estimated* hours for
  in-flight ones, so ten kernels pushed before any is collected cannot all pass
  a 30-hour ceiling on zero hours counted.
- **session** — one run blows it on its own. Kaggle stops the kernel at the cap
  and returns what it has. A 20-hour training run does not get 20 hours; it gets
  12 and a dead kernel.

```bash
python -m tools.kaggle quota --json          # hours left, per pool
python -m tools.kaggle accelerators --json   # ids, pools, session caps
```

Exit 13 is not exit 6, and the fixes are different. Waiting for the window to
roll is a real answer; so is a smaller run, or checkpointing and resuming across
several submissions. Raising `kaggle.quota.*` past what Kaggle actually allows is
not — Kaggle stops the kernel either way, and the only thing a higher local
ceiling buys is finding out mid-run.

## What a spec must declare

```toml
entrypoint = "train.py"
image      = "org/img@sha256:..."   # hashed, but Kaggle runs its own image
metrics_file = "metrics.json"

[estimate]
hours = 4.0          # REQUIRED here. See below.

[target]
accelerator = "NvidiaTeslaP100"   # optional; else [kaggle] default_accelerator
internet    = false               # optional; default off
dataset_sources = ["user/dataset"]      # for anything too big to pack
competition_sources = []
```

`[estimate] hours` is **required on this backend** and the submitter refuses
without it. On a dollar backend an absent estimate is merely optimistic —
`collect` replaces it with the platform's accounting and the ledger
self-corrects. Here the estimate is the only number the weekly pool has for a run
until it is collected, so a spec that declares none is a run the ceiling cannot
see. It does not have to be accurate, only present and honest.

`image` is still hashed and still must be digest-pinned, but Kaggle runs *its*
container, not yours. The image is part of the submission identity, not a
description of the environment — which is exactly why the smoke run matters more
here than anywhere else.

## One uploadable file

`kaggle kernels push` uploads the single file named in `kernel-metadata.json`
and nothing else. There is no `scp -r`. So the whole spec directory is packed
into a base64 blob **inside** a generated notebook whose first cell unpacks it
into `/kaggle/working`, and the entrypoint runs from there as a subprocess.

Consequences:

- **Keep the pipeline directory small.** It is capped, and the submitter refuses
  before uploading rather than letting Kaggle reject it. Bulk goes in a Kaggle
  dataset and is named in `[target] dataset_sources`.
- `__pycache__`, `.git`, `.venv`, `.pytest_cache` and `.ipynb_checkpoints` are
  never packed. Everything else beside the spec is, exactly as `scp -r` would.
- The generated notebook writes `grad_status.json` with the entrypoint's exit
  code *before* it fails the cell, so a failed run still says why.

## No internet by default

`enable_internet` is off unless a spec asks for it. A kernel without internet
cannot `pip install`, so the pipeline must run on Kaggle's preinstalled package
set — which is large, and pinned to their image rather than to your lockfile.
This is the single most common way a run that passed a local dry run fails on
Kaggle, and it is what `--smoke` exists to catch:

```bash
python -m tools.kaggle submit --spec <spec> --smoke --json
```

Turning internet on is a real trade, not a formality: a kernel that can reach the
network can also send anything it can read anywhere. Turn it on per spec, never
in `config/grad.toml`.

## What a run does

1. `submit` runs the four gates, then the session and weekly quota checks, then
   writes the in-flight run record holding its estimated hours against the pool,
   then packs and pushes the kernel.
2. `status` reads Kaggle's own kernel status. An output it does not recognise is
   reported as `unknown`, never guessed as `complete`.
3. `collect` downloads every output file, reads the marker for the exit code and
   the kernel log for the execution time, records `cost_usd_actual = 0.0` and the
   **actual** accelerator hours, and computes deviations mechanically.

Hours come from the kernel log where it is readable — queue time is Kaggle being
busy and is not charged to your allowance. Where it is not, `collect` falls back
to wall clock since submit and says so in `accelerator_hours_basis` rather than
quietly claiming precision it does not have.

The kernel is **not** deleted after collection, unlike an SSH working directory:
it is a durable, addressable record of the run with its own URL. Pass
`--delete-kernel` to opt into removing it.

## Credentials

Two halves, stored differently because only one of them is secret.

```bash
python -m tools.kaggle account --set <your-kaggle-username> --json
python -m tools.jobs credential set kaggle_key
python -m tools.kaggle account --check --json
```

`account --check` makes one authenticated call. A wrong username and a wrong key
fail it identically, which is fine — they have the same fix, which is to look at
the pair. `account` with no flags reports which account is in effect and where it
came from; `--clear` forgets it.

The account is stored under the app directory, not in `config/grad.toml` — that
file is hand-annotated and cannot be written back without discarding its
comments. `[kaggle] username` is still read when nothing is stored, and a stored
selection wins over it and says so.

Bare `kaggle` is denied by the hook, for the same reason bare `hf` is. The
credential never enters the agent's environment — `kaggle.py` puts it into one
subprocess's environment at the moment of use, and `credentials.scrub_environment`
removes `KAGGLE_USERNAME`, `KAGGLE_KEY` and `KAGGLE_CONFIG_DIR` at startup.
