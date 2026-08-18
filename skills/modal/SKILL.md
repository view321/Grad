---
name: modal
description: Modal's per-second GPU backend — H100s and up, how a spec becomes a Sandbox, why results come back through a Volume, and the 24-hour ceiling that cannot be raised. Load before the first Modal run.
---

# Modal sandboxes

`modal.py` is the fourth submitter and the first whose billing model the §6
dollar ceilings were already the right instrument for. Modal charges **per
second** against a published rate table, so `[spend]` is the gate here and there
is no second allowance to exhaust: **exit 13 never comes from this backend**.

That makes it the opposite of Kaggle in the way that matters for planning. On
Kaggle a run is free and the hours are scarce; here the hours are yours and the
money is the constraint. `python -m tools.modal ceilings --json` is the question
to ask before sizing a run.

## The hardware

```bash
python -m tools.modal gpus --json      # what is priced, and at what
```

| GPU | $/hour | note |
|---|---|---|
| T4 | 0.59 | |
| L4 | 0.80 | |
| A10G | 1.10 | |
| L40S | 1.95 | |
| A100-40GB | 2.10 | |
| A100-80GB | 2.50 | |
| H100 | 3.95 | the default |
| H200 | 4.54 | |
| B200 | 6.25 | |

A count suffix multiplies the rate: `H100:8` is eight cards at eight times the
price. **An accelerator absent from `[modal.gpu_rates]` is refused at submit**,
not booked at zero — the spend ceiling is the only gate here, and a ceiling that
cannot price a run is not bounding it.

## The 24-hour ceiling

Modal kills a Sandbox at 24 hours, whatever it was doing. `submit` refuses a
spec whose `[estimate] hours` (times `timeout_margin`, default 1.25) would need
longer, rather than starting a run that cannot finish. This is not a local
policy that can be raised: the container is stopped either way, and a higher
local number only changes when you find out.

A run that genuinely needs longer wants checkpointing and resuming across
several submissions — the same answer as a Kaggle session cap.

## What a spec must declare

```toml
entrypoint = "train.py"
image = "nvcr.io/nvidia/pytorch@sha256:..."   # digest, not a tag
argv = ["--config", "config.toml"]
metrics_file = "metrics.json"

[target]
platform = "modal"
gpu = "H100"                 # optional; [modal] default_gpu otherwise

[estimate]
hours = 3.0                  # required: it sets the sandbox timeout
cost_usd = 12.0
```

The image must be **digest-pinned**, exactly as on HF Jobs, and for the same
reason: `core/submission.py` hashes the digest, and the preflight record is keyed
by that hash. Modal's own image DSL (`pip_install`, `run_commands`) is
deliberately not reachable from a spec — an image assembled from a Python
expression has no digest until it is built, and a preflight keyed by a hash that
does not cover the environment certifies nothing.

## How your code gets there, and how results come back

The spec's directory is copied into the image at `/grad/pipeline` and the
entrypoint runs there. Nothing is uploaded separately and there is no notebook
to pack.

**Results come back through a Modal Volume, not the container filesystem.** A
sandbox's disk is gone the moment it exits, and `collect` runs afterwards by
construction — so a metrics file written beside the entrypoint would be
unreadable by the time anyone looked. Write to `$GRAD_METRICS_FILE`, which
points into the mounted Volume at `/grad/out/<run_id>/`. `$GRAD_OUT_DIR` is the
same directory, for checkpoints and figures worth keeping.

A pipeline that ignores `$GRAD_METRICS_FILE` and writes `metrics.json` in its
working directory is still collected — the wrapper copies it into the Volume
afterwards — but anything *else* it wrote is lost. Put it in `$GRAD_OUT_DIR`.

## Credentials

A token **pair**, both halves secret, both in the OS credential store:

```bash
python -m tools.jobs credential set modal_token_id
python -m tools.jobs credential set modal_token_secret
python -m tools.modal account --check --json     # does the pair authenticate
```

`--check` is the useful command. `credential status` says a secret is *stored*,
which is a different claim from a secret that *works*, and the gap between them
is otherwise discovered at the worst moment.

## The loop

```bash
python -m tools.preflight run --spec pipeline/spec.toml --json
python -m tools.ledger expect --task speedrun --quantity val_loss@1e9_tokens \
    --low 3.1 --high 3.4 --basis 'modded-nanogpt|README|3.28|8xH100' --json
python -m tools.modal submit --spec pipeline/spec.toml --expect exp-... --json
python -m tools.wakeup arm --run run-... --timeout 14400 --note 'nanogpt speedrun' --json
# end the turn; you will be woken
python -m tools.modal collect run-... --json
```

`collect` is non-blocking by default and exits 10 while the sandbox is still
running. Do not poll it in a loop — arm a wakeup and end the turn.

## What the cost number means

`collect` prices **wall clock from submission** against the rate table, bounded
by the sandbox's own timeout. That is an upper bound, not Modal's billing: it
includes the image pull and any delay between the run finishing and being
collected. Every Modal run carries a `cost_warning` saying so, and `cost_basis`
on the record is `wall_clock` rather than `measured`.

Collect promptly if the number matters. A wakeup is what makes that automatic.
