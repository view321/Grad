---
name: hf-jobs
description: Hugging Face Jobs patterns - flavors, images, artifacts, cost accounting, and the smoke path. Load before the first HF job or when a submission is refused upstream.
---

# Hugging Face Jobs

`jobs.py` is the only path in the system that can authenticate to HF. The token
lives in Windows Credential Manager and is read at the moment of use:

```bash
python -m tools.jobs credential set hf_token
python -m tools.jobs credential status --json
```

## Images

Pin by **digest**, never by tag. `preflight` refuses a tag it cannot resolve,
and this is not pedantry: `:latest` is exactly how the remote environment drifts
past a hash that otherwise looks airtight, which turns a passing preflight into
a false statement about the run that follows it.

```bash
docker manifest inspect --verbose myorg/train:2026-08 | jq -r .Descriptor.digest
```

## Flavors and cost

`config/grad.toml` holds the rate table under `[hf.flavor_rates]`. `collect`
multiplies the platform's own start/end timestamps by that rate — it never
reuses the estimate, because reusing the estimate would make the ceiling
self-fulfilling.

Keep the table current. A rate that is stale in the optimistic direction turns
the monthly ceiling into decoration.

## Artifacts

HF Jobs have no artifact channel of their own. Declare where the pipeline
uploads:

```toml
[config]
artifact_repo      = "myorg/run-artifacts"
artifact_repo_type = "dataset"
```

`collect` snapshots that repo into `ledger/runs/<run_id>/` and parses the
metrics file from it. Without it, only the job logs come back.

## The smoke path

```bash
python -m tools.jobs submit --spec pipeline/spec.toml --smoke --json
```

Exempt from the preflight and expectation gates — it runs *before* either can
exist for the submission it validates — and hard-capped in code instead: one
step, minutes of wall clock, cents, no artifact upload. Its spend still lands in
`runs.jsonl` and still counts toward the monthly ceiling. The caps are what keep
the exemption from becoming the way real jobs escape the gate.

Its result is written into the pending preflight record for the submission hash,
which is why the image digest is part of that hash.

## Typical sequence

```bash
python -m tools.preflight run --spec pipeline/spec.toml --only tests,dry_run --json
python -m tools.jobs submit --spec pipeline/spec.toml --smoke --json
python -m tools.ledger expect --task scaling-w2 --quantity val_loss@1e9_tokens \
    --low 2.9 --high 3.2 --basis 'arXiv:2001.08361|Fig 3|3.05|1.3B params' \
    --comparability 'our tokenizer differs; eval is a 5k held-out subset' --json
python -m tools.jobs submit --spec pipeline/spec.toml --expect exp-... --json
python -m tools.jobs collect run-... --json
python -m tools.ledger verdict run-... --quantity val_loss@1e9_tokens --verdict real --note '...' --json
```

## Failures worth knowing

- **401/403 on submit** — token scope. `credential status` tells you whether one
  is stored, not whether it is sufficient.
- **Job starts and dies immediately** — almost always the image missing a
  package the local venv happens to have. This is the exact failure `smoke`
  exists to catch for cents, so if you skipped it, run it now.
- **`metrics_missing` at collect** — the pipeline did not write the metrics
  artifact, or wrote it somewhere `artifact_repo` does not cover.
