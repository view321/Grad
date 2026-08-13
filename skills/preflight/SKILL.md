---
name: preflight
description: What each pre-flight check means, how to write a submission spec, and how to fix a failing gate. Load when a submitter refuses or when setting up a new pipeline.
---

# Pre-flight

The gate exists because remote-job failures are overwhelmingly boring — a shape
mismatch, a missing dependency, a bad path, a config that OOMs at step 0 — and
they are boring at the price of a GPU-hour. Every one of them is catchable in
advance for cents.

## The submission spec

One TOML file per pipeline, next to the entrypoint.

```toml
entrypoint   = "train.py"
argv         = ["--config", "configs/base.yaml"]
image        = "myorg/train@sha256:9f2c..."   # digest, never a tag
lockfile     = "requirements.lock"
config_file  = "configs/base.yaml"
metrics_file = "metrics.json"
source_roots = ["."]                          # where first-party imports resolve
extra_hash_paths = ["tokenizer/vocab.json"]   # runtime-loaded files

[dataset]
name     = "org/dataset"
revision = "a1b2c3d"                          # without this the data can change under the hash

[target]
platform = "hf"                               # or "ssh"
flavor   = "a10g-large"                       # or host = "gpu-box"

[estimate]
hours              = 3.5
rate_usd_per_hour  = 1.50
smoke_cost_usd     = 0.05

[config.checks]                               # optional, pipeline-declared
shapes     = "pytest -q tests/test_shapes.py"
grads      = "python tools/check_grads.py"
symbolic   = "python tools/check_derivation.py"
invariants = "pytest -q tests/test_invariants.py"

[config.dry_run]
argv = ["--steps", "1", "--batch-size", "2", "--max-samples", "10"]
```

## What the hash covers

The **resolved submission**, not a directory: the entrypoint and every
first-party module it imports (by import graph), the config *after* `--set`
overrides, the lock file, the dataset revision, the image digest, and argv.
There is no TTL — nothing decays by sitting still; what invalidates a record is
state change, and that is what a hash notices.

Two gaps, handled explicitly rather than silently: dynamic imports
(`importlib.import_module`) are invisible to static resolution, and files read
at runtime outside the config system are reached by neither the import graph nor
the config. Both go in `extra_hash_paths`. `preflight hash` prints the resolved
document, and its `warnings` array names what it could not see.

## The checks

| check | catches | note |
|---|---|---|
| `tests` | regressions in pipeline code | `pytest` on the pipeline's own tests |
| `dry_run` | logic, shape, config errors | local, seconds, free. Proves internal coherence and nothing more |
| `smoke` | everything local cannot see | **the highest-value check**: real image, real driver stack, real data path, real per-device batch size |
| `shapes` | tensor rank/axis errors | declare a command; `einops`/`jaxtyping` assertions on the dry run |
| `grads` | wrong hand-written gradients | `torch.autograd.gradcheck` against finite differences |
| `symbolic` | code drifting from the derivation | SymPy: differentiate the loss and compare to the implemented gradient |
| `invariants` | silent correctness bugs | `hypothesis`: seed determinism, equivariance, loss ≥ 0, densities normalise |
| `cost` | surprise bills | this job against the per-job ceiling, and the rolling 30-day total against the monthly one |

The local dry run does **not** catch a missing dependency in the remote image, a
CUDA/torch mismatch, distributed init, an unstaged dataset, a credential scope
problem, or OOM at the real batch size. That is what `smoke` is for. Run smoke at
the real per-device batch size with a truncated sequence count — at batch 2 it
does not test the thing that most often kills the real run.

## Fixing a refusal

| exit | meaning | do this |
|---|---|---|
| 4 | no passing preflight for this hash | `python -m tools.preflight run --spec <spec> --json` |
| 5 | no open expectation | `python -m tools.ledger expect --task ... --json`, then submit with `--expect` |
| 6 | spend ceiling | collect in-flight runs so their estimates become actuals, or raise the ceiling deliberately |
| 7 | stale uncollected run | `python -m tools.jobs collect <run_id> --json` |
| 9 | a check failed | read the log path in the error's `detail` |

If a gate fires and the reason looks wrong, that is worth investigating rather
than working around: the hash changed for a reason, and the reason is visible in
`preflight hash --spec <spec>`.
