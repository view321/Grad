"""grad-kaggle -- submit, watch, and collect Kaggle kernels (HANDOFF §6, §7, §9).

The third submitter, and the first whose scarce resource is not money. Kaggle
runs notebooks on a P100, a T4 pair, an A100, an H100 or a TPU for free, and
rations them by the hour: roughly 30 GPU h and 20 TPU h a week, and a single
session is stopped at 12 h (9 h on TPU).

So every §6 gate still runs -- preflight, expectation, spend, stale -- and the
spend gate passes every time, because $0.00 is under every ceiling. That is not a
gate doing its job quietly; it is a gate measuring the wrong quantity. What
bounds this backend is `core/kaggle_quota.py`, checked here in the same place and
in the same breath as the other four, and refusing with its own exit code (13)
so "you are out of GPU hours until Thursday" is never read as "you are out of
money".

**One uploadable file.** `kaggle kernels push` uploads the code file named in
`kernel-metadata.json` and nothing else -- there is no `scp -r` here. So the
pipeline is *embedded*: `_notebook_for` packs the whole spec directory into a
base64 blob inside a generated notebook whose first cell unpacks it. The set that
gets packed is the set the submission hash covers, which is what keeps "the
remote sees exactly what was preflighted" true on a backend that cannot copy a
directory.

**No general remote-execution capability (§9).** The Kaggle API key lives in
Windows Credential Manager, is read at the moment of use, and is put into one
subprocess's environment -- never into the agent's. `hooks.py` denies bare
`kaggle` for the same reason it denies bare `hf`: it is the speed bump, and the
credential is the wall.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from core import (
    budget,
    config as config_mod,
    credentials,
    gates,
    kaggle_quota,
    ledger_store as ls,
    submit as submit_lib,
)
from core.cli import Cli, main
from core.config import Config
from core.errors import (
    EXIT_RUNNING,
    ConfigError,
    GradError,
    UpstreamError,
    UsageError,
)
from core.submission import Submission, parse_override

cli = Cli(
    "grad-kaggle",
    "Submit and collect Kaggle kernels on free GPU/TPU accelerators.",
    epilog=(
        "Kaggle costs nothing and is rationed in hours, so the dollar ceilings pass every\n"
        "submission here and the gate that actually bounds this backend is the weekly\n"
        "accelerator allowance. It refuses with exit 13, never 6.\n\n"
        "  python -m tools.kaggle account --set <username> --json   # whose kernels these are\n"
        "  python -m tools.kaggle account --check --json            # does the pair authenticate\n"
        "  python -m tools.kaggle quota --json        # what is left this week\n"
        "  python -m tools.kaggle accelerators --json # what may be asked for\n\n"
        "The account name is stored as ordinary state and is readable; the API key is a\n"
        "credential: python -m tools.jobs credential set kaggle_key\n\n"
        "A spec must declare `[estimate] hours` (or duration_s) to be submitted here: the\n"
        "allowance is counted in hours, and a run that estimates none is a run the\n"
        "ceiling cannot see."
    ),
)

PLATFORM = "kaggle"
MARKER = "grad_status.json"

#: Directories never worth uploading, and one of them is a correctness issue
#: rather than a size one: a stale `__pycache__` from the developer's own
#: architecture shadows the sources it was built from.
_EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv", ".pytest_cache"}

#: Kaggle rejects an oversized kernel source, and it does so after the upload
#: with a message about the notebook rather than about the file that bloated it.
#: Refusing locally names the payload and points at the mechanism meant for bulk.
MAX_PAYLOAD_B64 = 1_000_000

#: How far past `MAX_PAYLOAD_B64` the *raw* bytes may go before `_payload_b64`
#: refuses without packing. Only a bound on how much this is willing to hold in
#: memory to find out; the real limit is checked on the finished blob.
_PACK_MEMORY_SLACK = 20

#: Filenames and suffixes that are credentials rather than pipeline. The whole
#: spec directory is uploaded, so this is the list that keeps §9's "no general
#: remote-execution capability" from being undone by a `.env` left beside the
#: entrypoint. Matched on the name, because that is what these are recognised by
#: -- a private key is not identifiable from its bytes.
_SECRET_NAMES = frozenset({
    ".env", ".netrc", "_netrc", ".npmrc", ".pypirc", ".htpasswd",
    "kaggle.json", "credentials.json", "credentials", "secrets.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
})
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk")
_SECRET_PREFIXES = ("client_secret", "service-account", "serviceaccount", ".env.")
#: Directories that are nothing but credentials. Refused rather than skipped,
#: like the files -- see `_payload_b64`.
_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".gcloud", ".azure", ".docker"})


def is_secret(rel: Path) -> bool:
    """Does this path, relative to the spec directory, look like a credential?

    Deliberately generous about what counts. A false positive is a file someone
    moves one directory up; a false negative is a key on Kaggle.

    Case-folded on both halves, and the directory half is the one that needed
    saying. This ships to Windows, where `.SSH` and `.ssh` are the same directory
    on disk but `Path.parts` hands back whatever casing was typed -- so comparing
    those raw let `.SSH/config` through while catching `.ssh/config`, which is
    the difference between two spellings of one directory and not a difference
    anyone would think to check.
    """
    if any(part.lower() in _SECRET_DIRS for part in rel.parts[:-1]):
        return True
    name = rel.name.lower()
    return (
        name in _SECRET_NAMES
        or name.endswith(_SECRET_SUFFIXES)
        or name.startswith(_SECRET_PREFIXES)
    )

_STATUSES = (
    "complete",
    "error",
    "cancelAcknowledged",
    "cancelRequested",
    "running",
    "queued",
)
_TERMINAL = {"complete", "error", "cancelAcknowledged"}


# ---------------------------------------------------------------------------
# the kaggle CLI
# ---------------------------------------------------------------------------
def _executable() -> str:
    """The `kaggle` beside *this* interpreter, then the one on PATH.

    The order matters more here than for a missing tool. `shutil.which` alone
    found the CLI in the user-site Python rather than in the venv, so this
    project's pinned `kaggle` was installed and a *different* installation's was
    what actually ran -- silently, against an API whose contract this module
    encodes. See `core/spawn.py:console_script`.
    """
    from core import spawn  # noqa: PLC0415

    found = spawn.console_script("kaggle")
    if found:
        return found
    raise ConfigError(
        "the `kaggle` CLI is not installed in this environment or on PATH, so Kaggle "
        "kernels cannot be reached",
        fix="pip install -e '.[kaggle]'",
    )


# ---------------------------------------------------------------------------
# which account
# ---------------------------------------------------------------------------
# A file rather than an entry in `config/grad.toml`, for the reason
# `budget.set_current` gives about the selected project: a command has to be
# able to write it, and `config/grad.toml` is a hand-annotated file that
# `tomllib` can read and cannot write -- rewriting it from a CLI would mean
# reformatting it and losing every comment in it.
#
# Under the app directory rather than the workspace, because a Kaggle account is
# a property of whoever installed this, exactly like the API key it pairs with,
# and not of the research sitting in one folder.
def account_path() -> Path:
    from core import appdata  # noqa: PLC0415 - import cycle if hoisted

    return appdata.state_dir() / "kaggle.json"


def stored_username() -> str | None:
    """The account `kaggle account --set` chose, or None."""
    from core import jsonl  # noqa: PLC0415

    record = jsonl.read_json(account_path()) or {}
    return str(record.get("username") or "").strip() or None


def validate_username(name: str) -> str:
    """A Kaggle username, or a refusal naming what is wrong with it.

    The slash matters more than it looks: a kernel reference is
    `<username>/<slug>`, so a value containing one silently re-points every push,
    status and collect at a path that is not this account's.
    """
    name = (name or "").strip()
    if not name:
        raise UsageError(
            "a Kaggle username cannot be empty",
            fix="python -m tools.kaggle account --set <your kaggle username> --json",
        )
    if "/" in name or any(c.isspace() for c in name):
        raise UsageError(
            f"{name!r} is not a Kaggle username: it contains a slash or whitespace, and a "
            "kernel reference is <username>/<slug>",
            fix="use the name shown on your Kaggle profile URL, e.g. kaggle.com/<this>",
        )
    return name


def resolve_username(cfg: Config) -> tuple[str | None, str]:
    """The account in effect, and where it came from.

    The stored selection wins over `[kaggle] username`, because a user who has
    just run `account --set` is entitled to expect it to take effect -- a command
    that silently does nothing because a config file disagrees is worse than one
    that overrides it. The command says so when it is shadowing a config value,
    so the precedence is never something you have to infer.
    """
    stored = stored_username()
    if stored:
        return stored, "state"
    configured = str(cfg.get("kaggle", "username", "") or "").strip()
    if configured:
        return configured, "config"
    return None, "unset"


def _username(cfg: Config) -> str:
    name, _ = resolve_username(cfg)
    if not name:
        raise ConfigError(
            "no Kaggle account is set, so no kernel slug can be built",
            fix="python -m tools.kaggle account --set <your kaggle username> --json",
        )
    return name


def _env(cfg: Config) -> dict[str, str]:
    """The credential pair, for the lifetime of one subprocess (§9).

    Built fresh per call and never cached: the key is read from the credential
    store at the moment of use, put into one child's environment, and is not in
    this process's own environment before or after. `credentials.scrub_environment`
    removes both variables from the agent at startup, so this function is the
    only way either reaches a `kaggle` process.
    """
    key = credentials.get(credentials.KAGGLE_KEY)
    if not (key or "").strip():
        raise ConfigError(
            f"credential {credentials.KAGGLE_KEY!r} is empty, so the Kaggle API cannot authenticate",
            fix=f"python -m tools.jobs credential set {credentials.KAGGLE_KEY}",
        )
    return {
        **os.environ,
        "KAGGLE_USERNAME": _username(cfg),
        "KAGGLE_KEY": key.strip(),
    }


def _run(argv: list[str], cfg: Config, *, timeout: float) -> str:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=_env(cfg)
        )
    except FileNotFoundError as exc:
        raise UpstreamError(
            f"{argv[0]} is not on PATH", fix="pip install -e '.[kaggle]'"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpstreamError(
            f"the kaggle CLI timed out after {timeout}s",
            fix="check connectivity to kaggle.com, or raise the matching kaggle.*_timeout_s in config/grad.toml",
        ) from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise UpstreamError(
            f"kaggle {argv[1] if len(argv) > 1 else ''} failed (exit {proc.returncode}): "
            f"{output.strip()[:400]}",
            fix="check the stored kaggle_key and that [kaggle] username matches the account it belongs to",
            detail={"argv": argv[1:], "output": output.strip()[:2000]},
        )
    return output


# ---------------------------------------------------------------------------
# staging: one uploadable file
# ---------------------------------------------------------------------------
def _add_bytes(tar: Any, rel: str, data: bytes) -> None:
    """One in-memory file into the tar, with the same determinism as the rest.

    The name is checked here rather than at the caller for the reason
    `tools/gpu.py:_write_remote` checks its own: this is content going onto a
    machine we do not own, and a relative path that climbs is worth refusing
    where it is written rather than trusting every future caller.
    """
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise UsageError(
            f"refusing to pack {rel!r}: a payload entry is a path inside the pipeline",
            fix="pass a name relative to the spec directory, with no '..' in it",
        )
    info = tarfile.TarInfo(rel)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def _payload_b64(
    sub: Submission, *, overrides: dict[str, str] | None = None
) -> tuple[str, list[str]]:
    """The spec directory, packed into one base64 blob.

    The whole directory rather than only the import graph, so this backend stages
    what `gpu.py`'s `scp -r` stages. A pipeline that reads a CSV sitting beside
    its entrypoint works on an SSH host and would fail here on a narrower rule,
    and "it ran on the other backend" is the least useful bug report there is.

    `overrides` replaces or adds files by relative path on the way into the tar,
    without touching the directory on disk. That is what an evolve candidate is:
    the preflighted pipeline with one program swapped for a mutated one. Doing it
    here rather than by copying the tree to a temp directory keeps one packing
    implementation -- the secret scan, the size refusals and the deterministic
    mtimes all apply to a candidate exactly as they do to a submission, which
    they would not if candidates got a packer of their own.
    """
    base = sub.spec_path.parent
    eligible: list[Path] = []
    secrets: list[str] = []
    raw_bytes = 0
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        if not path.is_file():
            continue
        if is_secret(rel):
            secrets.append(rel.as_posix())
            continue
        eligible.append(path)
        try:
            raw_bytes += path.stat().st_size
        except OSError:  # counted as nothing; the pack below will raise on it
            continue
    # Named and refused, not skipped. This blob leaves the machine, and §9's rule
    # is that a credential goes into one subprocess's environment at the moment
    # of use and nowhere else -- a `.env` sitting beside the entrypoint would
    # ride to Kaggle inside the notebook source, private kernel or not. Silently
    # dropping it would be worse than either: the run would fail remotely for a
    # file that is right there locally, which is the bug report `_payload_b64`'s
    # docstring says this backend exists to avoid.
    if secrets:
        raise UsageError(
            f"the pipeline directory holds {len(secrets)} file(s) that look like "
            f"credentials, and this whole directory is uploaded: {', '.join(secrets[:5])}",
            fix=(
                "move them outside the spec directory -- a Kaggle kernel's source is public to "
                "anyone the kernel is shared with, and the API key belongs in Windows "
                "Credential Manager (python -m tools.jobs credential set kaggle_key)"
            ),
        )
    # Refused on the way in, not after packing: a spec directory with a 4 GB
    # checkpoint in it used to be gzipped and base64-encoded *entirely in
    # memory* before anything said no -- a refusal costing more than the run.
    #
    # Against a multiple of the limit rather than the limit itself, because the
    # units differ: this counts raw bytes and `MAX_PAYLOAD_B64` bounds the
    # base64 of a gzip. The implication only runs one way -- raw bytes that fit
    # certainly pack small enough, but raw bytes that do not may still compress
    # under it, and a tree of generated source is exactly that case. So this is
    # a memory guard, deliberately loose, and `len(blob)` below is still the
    # check that decides. A directory would have to beat 26:1 to be refused here
    # and accepted there, which is a file of zeros rather than a pipeline.
    if raw_bytes > MAX_PAYLOAD_B64 * _PACK_MEMORY_SLACK:
        raise UsageError(
            f"the pipeline directory holds {raw_bytes:,} bytes of files, too much to pack "
            f"in memory against a {MAX_PAYLOAD_B64:,}-byte kernel source limit",
            fix=(
                "move the bulk out of the spec directory and reference it as a Kaggle dataset "
                "in the spec's [target] dataset_sources -- kernel sources are for code"
            ),
        )
    packed: list[str] = []
    buffer = io.BytesIO()
    # `gzip` rather than `tar` alone, and deterministically: mtime=0 keeps the
    # blob byte-identical between two pushes of an unchanged directory, which is
    # what makes a re-push diffable.
    replacements = {str(k): str(v) for k, v in (overrides or {}).items()}
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as tar:
        for path in eligible:
            rel = path.relative_to(base).as_posix()
            if rel in replacements:
                # Replaced, not appended alongside. Two entries with one name in
                # a tar is a file whose contents depend on extraction order,
                # which for a candidate means a score that depends on tar.
                _add_bytes(tar, rel, replacements.pop(rel).encode("utf-8"))
                packed.append(rel)
                continue
            info = tar.gettarinfo(str(path), arcname=rel)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
            packed.append(rel)
        # Whatever the overrides added rather than replaced. Sorted so the blob
        # stays a function of its inputs, like the sorted walk above.
        for rel in sorted(replacements):
            _add_bytes(tar, rel, replacements[rel].encode("utf-8"))
            packed.append(rel)
    blob = base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(blob) > MAX_PAYLOAD_B64:
        raise UsageError(
            f"the pipeline directory packs to {len(blob):,} base64 bytes, past the "
            f"{MAX_PAYLOAD_B64:,} a Kaggle kernel source can carry",
            fix=(
                "move the bulk out of the spec directory and reference it as a Kaggle dataset "
                "in the spec's [target] dataset_sources -- kernel sources are for code"
            ),
        )
    return blob, packed


def _notebook_for(sub: Submission, command: list[str], *, payload: str) -> dict[str, Any]:
    """Generate the .ipynb Kaggle will run, pipeline and all.

    Built as plain JSON rather than through `nbformat`: the v4 schema for a code
    cell is four keys, and a submitter whose only hard dependency is the `kaggle`
    CLI is one less thing to install on a machine that just wants to push a job.

    **The payload is a string literal inside cell 1, not a file beside it.**
    `kernels push` uploads the one file named in `code_file` and the metadata,
    and nothing else in the directory -- so a payload written next to the
    notebook is a payload that stays on this machine, and the kernel fails on
    Kaggle with a FileNotFoundError for a file that exists locally. Everything
    the run needs has to be inside this document.

    Three cells, in the order they have to be:

      1. unpack the pipeline into the working directory;
      2. run the entrypoint as a subprocess, so its exit code is a value rather
         than an exception's type;
      3. write the marker and *then* fail the cell if the exit code was non-zero.

    The order in cell 3 is the whole point. Kaggle's own status is the coarse
    signal (`complete` / `error`) and the marker is the precise one, so the
    marker has to be written before anything that can stop the cell -- otherwise
    the one run whose exit code matters most is the one that has no marker.
    """
    argv = json.dumps(command)
    # Chunked across source lines rather than written as one megabyte-long line:
    # a notebook is a document people open, and some editors and diff tools give
    # up on a single line that long.
    chunks = [payload[i : i + 4096] for i in range(0, len(payload), 4096)] or [""]
    source_unpack = [
        "import base64, io, tarfile\n",
        "blob = (\n",
        *[f"    {chunk!r}\n" for chunk in chunks],
        ")\n",
        "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob))) as t:\n",
        # `filter='data'` explicitly. Python 3.14 made it the default and 3.12
        # warns without it, but the kernel runs on Kaggle's image and not on this
        # machine's interpreter -- so the version that matters is one this
        # process cannot see. Named rather than inherited, and guarded by
        # `hasattr` rather than by catching `TypeError`: a bare except there
        # would also swallow a real error from inside the extraction and fall
        # back to the unfiltered behaviour, which is the one outcome worth
        # avoiding. On a 3.11 without the filter this extracts as it always did.
        "    if hasattr(tarfile, 'data_filter'):\n",
        "        t.extractall('/kaggle/working', filter='data')\n",
        "    else:\n",
        "        t.extractall('/kaggle/working')\n",
        "    print('unpacked', len(t.getnames()), 'files')\n",
    ]
    source_run = [
        "import json, subprocess, sys, time\n",
        f"argv = {argv}\n",
        "started = time.time()\n",
        "proc = subprocess.run(argv, cwd='/kaggle/working', capture_output=True, text=True)\n",
        "elapsed = time.time() - started\n",
        "open('/kaggle/working/stdout.log', 'w', encoding='utf-8').write(proc.stdout or '')\n",
        "open('/kaggle/working/stderr.log', 'w', encoding='utf-8').write(proc.stderr or '')\n",
        "sys.stdout.write((proc.stdout or '')[-20000:])\n",
        "sys.stderr.write((proc.stderr or '')[-20000:])\n",
    ]
    source_marker = [
        "marker = {'state': 'finished', 'exit_code': proc.returncode,\n",
        "          'elapsed_s': round(elapsed, 3)}\n",
        f"open('/kaggle/working/{MARKER}', 'w', encoding='utf-8').write(json.dumps(marker))\n",
        "print('grad_status', json.dumps(marker))\n",
        "# After the marker is on disk, never before: a failing run is exactly the\n",
        "# one whose exit code has to survive, and raising first would lose it.\n",
        "if proc.returncode != 0:\n",
        "    raise SystemExit(f'entrypoint exited {proc.returncode}')\n",
    ]

    def cell(source: list[str]) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    return {
        "cells": [cell(source_unpack), cell(source_run), cell(source_marker)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            # Not read by Kaggle; read by a human opening the kernel and asking
            # which submission this was.
            "grad": {"submission_hash": sub.hash(), "spec": str(sub.spec_path)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _slug(cfg: Config, run_id: str) -> str:
    """A Kaggle kernel slug: lowercase, hyphenated, and never with an underscore.

    The underscore is not a style choice. A title containing one makes the
    execution instance's status unobtainable through the API, so `collect` would
    poll a kernel it can never see finish -- the run goes stale and then blocks
    every later submission through the §6 stale gate.
    """
    prefix = str(cfg.get("kaggle", "kernel_prefix", "grad") or "grad")
    raw = f"{prefix}-{run_id}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", raw)).strip("-")


def _metadata(cfg: Config, sub: Submission, *, ref: str, slug: str, accelerator: str) -> dict[str, Any]:
    target = sub.target or {}
    private = bool(target.get("is_private", cfg.get("kaggle", "is_private", True)))
    internet = bool(target.get("internet", cfg.get("kaggle", "enable_internet", False)))
    return {
        "id": ref,
        # Title tracks the slug exactly, underscores and all (there are none).
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        # Both spellings. The `--accelerator` flag is the current mechanism and
        # these are the older booleans; Kaggle honours the flag, and a client
        # that reads only the metadata still sees the right shape.
        "enable_gpu": cfg.accelerator_kind(accelerator) == "gpu",
        "enable_tpu": cfg.accelerator_kind(accelerator) == "tpu",
        "enable_internet": internet,
        "dataset_sources": list(target.get("dataset_sources", [])),
        "competition_sources": list(target.get("competition_sources", [])),
        "kernel_sources": list(target.get("kernel_sources", [])),
        "model_sources": list(target.get("model_sources", [])),
    }


def _stage(cfg: Config, sub: Submission, workdir: Path, *, ref: str, slug: str,
           accelerator: str, command: list[str],
           overrides: dict[str, str] | None = None) -> list[str]:
    payload, packed = _payload_b64(sub, overrides=overrides)
    notebook = _notebook_for(sub, command, payload=payload)
    (workdir / f"{slug}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False), encoding="utf-8"
    )
    (workdir / "kernel-metadata.json").write_text(
        json.dumps(_metadata(cfg, sub, ref=ref, slug=slug, accelerator=accelerator), indent=2),
        encoding="utf-8",
    )
    return packed


# ---------------------------------------------------------------------------
# accelerator + hours resolution
# ---------------------------------------------------------------------------
def resolve_accelerator(flag: str | None, sub: Submission | None, cfg: Config) -> str:
    """`--accelerator` -> the spec's `[target] accelerator` -> the configured default.

    Mirrors how `jobs.py` resolves a flavor, so the two backends answer "where
    does this setting come from" the same way.
    """
    name = flag or (sub.target.get("accelerator") if sub else None) or cfg.get(
        "kaggle", "default_accelerator", "NvidiaTeslaP100"
    )
    cfg.accelerator_kind(str(name))  # raises ConfigError on an unknown id
    return str(name)


def require_platform(sub: Submission) -> None:
    """The spec must name this backend, or its preflight is about another one.

    `tools/preflight.py` picks which backend to smoke on by reading
    `[target] platform`, and gate 1 then accepts that record for *this*
    submission hash without rechecking where it came from. So a spec saying
    `platform = "hf"` submitted through this CLI passes gate 1 on a preflight
    that ran somewhere else entirely -- an A10G proving nothing about a P100,
    against a preinstalled package set Kaggle does not have.

    An absent platform is left alone: that is not a mismatch, it is a spec
    preflight will refuse to smoke at all, with its own instruction.
    """
    declared = str(sub.target.get("platform") or "").lower()
    if declared and declared != PLATFORM:
        raise UsageError(
            f"this spec's [target] platform is {declared!r}, so its preflight smoke ran on "
            f"{declared} -- a passing record from another backend says nothing about Kaggle's "
            "image, accelerator or package set",
            fix=f'set platform = "kaggle" under [target] in {sub.spec_path}, then re-run the preflight',
        )


def estimated_hours(sub: Submission) -> float:
    """The spec's estimated duration, in hours, or a refusal.

    A spec with no declared duration estimates 0.0 hours, and 0.0 hours passes
    every weekly allowance forever. On a dollar backend that is merely optimistic
    -- `collect` replaces the estimate with the platform's own accounting and the
    ledger self-corrects. Here it is the difference between a ceiling and a
    decoration, because until the run is collected the estimate is the *only*
    number the pool has for it. So it is required, in the same spirit as
    `check_smoke_caps` refusing a target whose rate it cannot look up.
    """
    try:
        hours = sub.estimated_duration_s() / 3600.0
    except (TypeError, ValueError) as exc:
        # `hours = "two"` in a spec would otherwise surface as exit 1, "a bug in
        # the CLI", for what is a typo in a file the agent wrote. Non-finite
        # values are *not* caught here on purpose: they pass this and are refused
        # by `kaggle_quota._finite_hours`, which is the one place that decides
        # what can be compared against an allowance.
        raise UsageError(
            f"the spec's [estimate] is not a number: {exc}",
            fix=f"fix [estimate] hours (or duration_s) in {sub.spec_path}",
        ) from exc
    if hours <= 0:
        raise UsageError(
            "this spec declares no estimated duration, so its accelerator hours cannot be "
            "counted against the weekly allowance -- and an allowance that cannot count a "
            "run is not an allowance",
            fix=(
                "add `[estimate] hours = <n>` (or duration_s) to "
                f"{sub.spec_path}; it does not have to be exact, and collect replaces it "
                "with what the kernel actually used"
            ),
        )
    return hours


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
def _submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True)
    p.add_argument(
        "--accelerator",
        help="Kaggle accelerator id (defaults to the spec's target.accelerator, then [kaggle] default_accelerator)",
    )
    p.add_argument("--expect", help="expectation id to bind. REQUIRED unless --smoke")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--task")
    p.add_argument("--project", help="project to charge this run to (defaults to the current one; §15)")
    p.add_argument("--smoke", action="store_true", help="gate-exempt, hard-capped one-step check (§6)")
    p.add_argument("--no-digest", action="store_true", help=argparse.SUPPRESS)


@cli.command("submit", "submit a kernel (gated) or a smoke check (capped)", setup=_submit_args)
def cmd_submit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    sub = Submission.load(
        args.spec,
        overrides=dict(parse_override(o) for o in args.overrides),
        resolve_digest=not args.no_digest,
    )
    require_platform(sub)
    accelerator = resolve_accelerator(args.accelerator, sub, cfg)
    kind = cfg.accelerator_kind(accelerator)
    project_id = budget.resolve(args.project)

    if args.smoke:
        if args.expect:
            raise UsageError("--smoke binds no prediction", fix="drop --expect, or drop --smoke")
        result = run_smoke(sub, cfg, accelerator=accelerator, project=project_id)
        from tools import preflight

        preflight.record_check_result(sub.hash(), "smoke", result)
        if not result.get("ok"):
            raise GradError(
                "smoke_failed",
                result.get("reason", "the smoke check failed on Kaggle"),
                exit_code=9,
                fix=result.get("fix") or "read the smoke log under ledger/runs/",
                detail=result,
            )
        return {"smoke": result, "submission_hash": sub.hash()}

    # The four gates first, then the fifth. Order is `check_submit`'s own
    # argument carried one gate further: a missing preflight is the most common
    # refusal and the most actionable, and it is true regardless of how many
    # hours are left in the week. The estimate is read in between because it is
    # the fifth gate's only input, not a gate of its own.
    summary = submit_lib.check(sub, args.expect, cfg, project=project_id)
    hours = estimated_hours(sub)
    quota = kaggle_quota.check(cfg, kind, hours, accelerator=accelerator)

    run_id, _ = submit_lib.record_submission(
        sub,
        expectation_id=args.expect,
        platform=PLATFORM,
        target={
            "platform": PLATFORM,
            "accelerator": accelerator,
            "kind": kind,
            # Zero, and recorded rather than omitted: a reader comparing this
            # record against an HF one should see the number, not its absence.
            "rate_usd_per_hour": 0.0,
        },
        command=_command_for(sub),
        task=args.task,
        project=project_id,
        extra={
            kaggle_quota.F_ACCELERATOR: accelerator,
            kaggle_quota.F_KIND: kind,
            # What the weekly pool counts this run at until it is collected.
            kaggle_quota.F_ESTIMATE: round(hours, 4),
        },
        cfg=cfg,
        # Re-checked inside the append lock, for the reason `cfg` is: the gate
        # above read the ledger and this run's hours land in it afterwards, so
        # two submitters racing -- the agent and a terminal, or the agent and a
        # UI-spawned task -- could both pass a 30-hour allowance with 20-hour
        # estimates against 0 used, and both commit. The dollar ceilings do not
        # get to be the only ones that survive a race.
        precondition=lambda: kaggle_quota.check(cfg, kind, hours, accelerator=accelerator),
    )

    slug = _slug(cfg, run_id)
    ref = f"{_username(cfg)}/{slug}"
    try:
        with tempfile.TemporaryDirectory(prefix="grad-kaggle-") as tmp:
            workdir = Path(tmp)
            packed = _stage(
                cfg, sub, workdir, ref=ref, slug=slug,
                accelerator=accelerator, command=_command_for(sub),
            )
            _push(cfg, workdir, accelerator=accelerator,
                  timeout_s=float(cfg.get("kaggle", "push_timeout_s", 900)))
    except GradError as exc:
        # The in-flight record already exists and is already holding hours
        # against the weekly pool. Left alone it would go stale and block every
        # later submission -- for a kernel that never started. `gpu.py` and
        # `jobs.py` both finalise the equivalent failure; neither backend gets to
        # differ on this.
        submit_lib.finish(
            run_id,
            status="submit_failed",
            results={},
            cost_usd_actual=0.0,
            artifacts_dir=submit_lib.artifacts_dir(run_id),
            expectation=None,
            extra={
                "error": exc.message,
                "kernel_ref": ref,
                # Zero, explicitly: the push failed, so the pool must give the
                # estimated hours back rather than hold them for a week.
                kaggle_quota.F_ACTUAL: 0.0,
            },
        )
        raise

    submit_lib.attach_handle(
        run_id,
        {"kernel_ref": ref, "slug": slug, "accelerator": accelerator, "kind": kind,
         "url": _url(ref)},
    )
    return {
        "run_id": run_id,
        "kernel_ref": ref,
        "url": _url(ref),
        "accelerator": accelerator,
        "kind": kind,
        "estimated_hours": round(hours, 4),
        "files_staged": len(packed),
        "gates": summary,
        "quota": quota,
        "next": f"python -m tools.kaggle collect {run_id} --json",
    }


def _command_for(sub: Submission) -> list[str]:
    if sub.target.get("command"):
        return [str(c) for c in sub.target["command"]]
    return ["python", sub.entrypoint.name, *sub.argv]


def _url(ref: str) -> str:
    """Where a human reads this run. Kaggle serves kernels under /code/."""
    return f"https://www.kaggle.com/code/{ref}"


def _push(
    cfg: Config,
    workdir: Path,
    *,
    accelerator: str,
    timeout_s: float,
    kernel_timeout_s: int | None = None,
) -> str:
    """Upload and start the kernel.

    `timeout_s` bounds *this call* -- `push` returns once Kaggle accepts the
    upload -- and is unrelated to how long the kernel then runs.
    `kernel_timeout_s` is the one that bounds the kernel, and it is what makes
    the smoke cap real rather than advisory: without it the only limit is how
    long we choose to poll, which stops us waiting and does not stop the job.
    """
    argv = [_executable(), "kernels", "push", "-p", str(workdir)]
    if accelerator and cfg.accelerator_kind(accelerator) != "cpu":
        argv += ["--accelerator", accelerator]
    if kernel_timeout_s:
        argv += ["--timeout", str(int(kernel_timeout_s))]
    return _run(argv, cfg, timeout=timeout_s)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def _parse_status(output: str) -> dict[str, Any]:
    """Read a kernel's status out of whatever the CLI printed.

    Tries JSON first in case a newer CLI emits it, then looks for a known status
    token. Deliberately conservative: an unrecognised output reports `unknown`
    rather than guessing at `complete`, because guessing `complete` collects a
    kernel that is still running and writes a run record with no results in it.
    """
    text = (output or "").strip()
    try:
        doc = json.loads(text)
        if isinstance(doc, dict) and doc.get("status"):
            return {"status": str(doc["status"]), "message": doc.get("failureMessage") or ""}
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    for status in _STATUSES:
        if status.lower() in lowered:
            return {"status": status, "message": text[:400]}
    return {"status": "unknown", "message": text[:400]}


def _status(cfg: Config, ref: str) -> dict[str, Any]:
    try:
        output = _run(
            [_executable(), "kernels", "status", ref],
            cfg,
            timeout=float(cfg.get("kaggle", "status_timeout_s", 120)),
        )
    except GradError as exc:
        return {"status": "unknown", "message": exc.message}
    return _parse_status(output)


@cli.command("status", "report a run's state without collecting it", setup=lambda p: p.add_argument("run_id"))
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    r = ls.run(args.run_id)
    handle = r.get("handle") or {}
    payload = {
        "run_id": r.id,
        "ledger_status": r.status,
        "collected": r.collected,
        "stale": ls.is_stale(r, cfg=cfg),
        "kernel_ref": handle.get("kernel_ref"),
        "accelerator": r.get(kaggle_quota.F_ACCELERATOR),
        "estimated_hours": r.get(kaggle_quota.F_ESTIMATE),
    }
    if handle.get("kernel_ref") and not r.collected:
        payload["kernel"] = _status(cfg, handle["kernel_ref"])
    return payload


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def run_smoke(
    sub: Submission, cfg: Config, *, accelerator: str | None = None, project: str | None = None
) -> dict[str, Any]:
    """One step on a real Kaggle accelerator, capped in code (§6).

    `accelerator` defaults rather than being required, because `tools/preflight.py`
    calls every backend's smoke as `run_smoke(sub, cfg)` and nothing else. A
    required argument here would make this backend submittable but not
    preflightable -- and since a passing `smoke` check is gate 1's input, every
    Kaggle job would then be refused for a reason that had nothing to do with it.

    The caps are computed against a rate of zero, which passes the cost clamp
    unconditionally -- on this backend the wall-clock cap is doing all the work,
    and it is the honest one to lean on because wall clock is what Kaggle
    rations. That cap is handed to Kaggle as the kernel's own `--timeout`, not
    merely used as a local poll deadline: a deadline this side of the network
    stops us *waiting*, it does not stop the kernel, and a smoke that keeps
    running after we walked away is the exemption turning into the hole in the
    allowance that §6 says it must not become.

    Smoke skips the four gates and the quota gate alike, matching `record_smoke_run`'s
    contract, but its hours are **recorded** and counted against the weekly pool by
    every later submission. Otherwise the exemption would be a hole in the allowance
    as well as in the gate, which is the exact argument §6 makes about the ceiling.
    """
    accelerator = accelerator or resolve_accelerator(None, sub, cfg)
    kind = cfg.accelerator_kind(accelerator)
    caps = gates.check_smoke_caps(
        sub, cfg, rate_usd_per_hour=0.0, target_name=f"Kaggle {accelerator}"
    )
    command = [*_command_for(sub), "--steps", str(caps["steps"]), "--smoke"]
    run_id = submit_lib.record_smoke_run(
        sub, cfg=cfg, platform=PLATFORM,
        target={"platform": PLATFORM, "accelerator": accelerator, "kind": kind, "rate_usd_per_hour": 0.0},
        caps=caps, command=command, project=project,
        extra={
            kaggle_quota.F_ACCELERATOR: accelerator,
            kaggle_quota.F_KIND: kind,
            kaggle_quota.F_ESTIMATE: round(float(caps["timeout_s"]) / 3600.0, 4),
        },
    )
    artifacts = submit_lib.artifacts_dir(run_id)
    slug = _slug(cfg, run_id)
    ref = f"{_username(cfg)}/{slug}"

    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="grad-kaggle-") as tmp:
            workdir = Path(tmp)
            _stage(cfg, sub, workdir, ref=ref, slug=slug, accelerator=accelerator, command=command)
            _push(
                cfg, workdir, accelerator=accelerator,
                timeout_s=float(cfg.get("kaggle", "push_timeout_s", 900)),
                # The §6 cap, enforced by Kaggle rather than by our patience.
                kernel_timeout_s=int(caps["timeout_s"]),
            )
        # Kaggle queues before it runs, and the queue is not part of the cap the
        # spec asked for -- so the poll waits the wall-clock cap *plus* a grace,
        # rather than treating queue time as smoke time.
        state = _wait(cfg, ref, deadline=time.time() + caps["timeout_s"] + _queue_grace(cfg))
    except GradError as exc:
        submit_lib.finish(
            run_id, status="failed", results={}, cost_usd_actual=0.0,
            artifacts_dir=artifacts, expectation=None,
            extra={"error": exc.message, "kernel_ref": ref, kaggle_quota.F_ACTUAL: 0.0},
        )
        return {"ok": False, "reason": exc.message, "fix": exc.fix, "run_id": run_id}

    marker, log_hours, log_text = _fetch_output(cfg, ref, artifacts)
    (artifacts / "smoke.log").write_text(log_text or "", encoding="utf-8")
    exit_code = marker.get("exit_code")
    hours = log_hours if log_hours is not None else (time.time() - started) / 3600.0
    # Three things have to be true, and the third is why the poll's own verdict
    # is kept: a kernel still queued when the grace ran out has no status worth
    # believing and no marker at all, and calling that a pass because nothing
    # said otherwise is how a smoke check certifies an environment it never
    # reached.
    timed_out = state.get("status") not in _TERMINAL
    ok = not timed_out and state.get("status") == "complete" and exit_code in (0, None)

    if not caps["artifact_upload"]:
        # The §6 carve-out's own rule, and it matters more here than on SSH: a
        # kernel is a durable, addressable object with its output attached, so a
        # smoke left behind is a set of results with a URL that outlives the
        # check it came from. `gpu.py` removes the remote directory for exactly
        # this reason; the kernel is the equivalent.
        try:
            _run([_executable(), "kernels", "delete", ref, "-y"], cfg,
                 timeout=float(cfg.get("kaggle", "status_timeout_s", 120)))
        except GradError:
            pass

    submit_lib.finish(
        run_id,
        status="completed" if ok else "failed",
        results={}, cost_usd_actual=0.0,
        artifacts_dir=artifacts, expectation=None,
        extra={
            "exit_code": exit_code, "smoke": True, "kernel_ref": ref,
            "kernel_status": state.get("status"),
            "timed_out": timed_out,
            kaggle_quota.F_ACTUAL: round(hours, 4),
        },
    )
    if timed_out:
        reason = (
            f"the smoke kernel was still {state.get('status')} after "
            f"{int(caps['timeout_s'] + _queue_grace(cfg))}s, so nothing was proved about the environment"
        )
        fix = (
            "Kaggle was queueing rather than running -- the kernel carries its own "
            f"{int(caps['timeout_s'])}s cap and will stop itself. Re-run the check, or raise "
            "kaggle.queue_grace_s in config/grad.toml if the queue is routinely this long"
        )
    else:
        reason = f"the smoke kernel finished {state.get('status')} (exit {exit_code})"
        fix = f"read {artifacts / 'smoke.log'} -- this is the environment the real job would have used"

    return {
        "ok": ok,
        "run_id": run_id,
        "kernel_ref": ref,
        "accelerator": accelerator,
        "exit_code": exit_code,
        "kernel_status": state.get("status"),
        "timed_out": timed_out,
        "accelerator_hours": round(hours, 4),
        "cost_usd": 0.0,
        "caps": caps,
        "log": str(artifacts / "smoke.log"),
        "output": "\n".join((log_text or "").splitlines()[-25:]),
        "reason": None if ok else reason,
        "fix": None if ok else fix,
        "scope": (
            "remote; exercises the real Kaggle image, the real accelerator, and the "
            "preinstalled package set a kernel cannot add to without internet"
        ),
    }


# ---------------------------------------------------------------------------
# evolve candidates
# ---------------------------------------------------------------------------
#: How many bytes of a candidate's output are kept, matching `tools/gpu.py`.
CANDIDATE_OUTPUT_BYTES = 8000


def evaluate_candidate(
    sub: Submission,
    cfg: Config,
    *,
    candidate_id: str,
    files: dict[str, str],
    command: list[str],
    timeout_s: int,
    artifacts: Path,
    accelerator: str | None = None,
) -> dict[str, Any]:
    """Run one evolve candidate as a Kaggle kernel, and read its metrics back.

    A candidate here is a changed architecture or a changed optimiser, so its
    evaluation is a training run and one kernel per candidate is the right unit
    -- the minute or two of queue and start-up is noise against it. This is the
    same push/poll/fetch the real submitter does, with two differences that both
    follow from a candidate not being a run.

    **No ledger row.** `tools/evolve.py` records candidates in
    `candidates.jsonl` so a long campaign cannot dominate a ledger read by hand
    (§23 item 4), and that holds when the search goes remote. The campaign is the
    ledgered unit and its expectation is the bound prediction.

    **The hours are still counted.** `core/kaggle_quota.py` folds candidate rows
    beside runs, because the weekly accelerator allowance is a hard external
    limit and a campaign that spent it invisibly would surface as an ordinary
    submission being refused for hours nothing could account for. The caller
    checks the projection before generation 0; this records what was actually
    used so the fold has something to read.

    The mutated program reaches the kernel the same way the pipeline does --
    inside the notebook's base64 payload, via `_payload_b64`'s `overrides`. There
    is no second delivery path, so the secret scan and the size refusals apply to
    a candidate exactly as they do to a submission.
    """
    accelerator = resolve_accelerator(accelerator, sub, cfg)
    kind = cfg.accelerator_kind(accelerator)
    slug = _slug(cfg, candidate_id)
    ref = f"{_username(cfg)}/{slug}"
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def _failed(message: str, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "exit_code": None,
            "output": "",
            "error": message,
            "hours": round((time.time() - started) / 3600.0, 4),
            "cost_usd": 0.0,
            "accelerator": accelerator,
            "accelerator_kind": kind,
            "where": _url(ref),
            **extra,
        }

    try:
        with tempfile.TemporaryDirectory(prefix="grad-cand-") as tmp:
            workdir = Path(tmp)
            _stage(
                cfg, sub, workdir,
                ref=ref, slug=slug, accelerator=accelerator, command=command,
                overrides=files,
            )
            _push(
                cfg, workdir,
                accelerator=accelerator,
                timeout_s=float(cfg.get("kaggle", "push_timeout_s", 600)),
                # Bounded where it runs, not only where it is watched. Without
                # this the only limit is how long we choose to poll, which stops
                # us waiting and does not stop the kernel -- and an abandoned
                # kernel keeps spending the weekly allowance.
                kernel_timeout_s=int(timeout_s),
            )
    except GradError as exc:
        return _failed(exc.message)

    state = _wait(cfg, ref, deadline=time.time() + int(timeout_s) + _queue_grace(cfg))
    if state.get("status") not in _TERMINAL:
        # The kernel's own timeout should have ended it. Reaching here means
        # Kaggle is queueing or not answering, so it is left alone rather than
        # guessed at -- `kernels status` is the only thing that knows.
        return _failed(
            f"the candidate was still {state.get('status') or 'unknown'} after "
            f"{int(timeout_s)}s plus the queue grace",
            kernel_status=state.get("status"),
        )

    marker, hours, log_text = _fetch_output(cfg, ref, artifacts)
    exit_code = marker.get("exit_code")
    if exit_code is None:
        # No marker means the kernel died before cell 3, which is a real
        # outcome and not an exit code. Reported as one would be a number
        # nobody measured.
        return _failed(
            f"the kernel finished as {state.get('status')} without recording an outcome",
            kernel_status=state.get("status"),
            hours=round(hours or (time.time() - started) / 3600.0, 4),
            output=log_text[-CANDIDATE_OUTPUT_BYTES:],
        )

    exit_code = int(exit_code)
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "output": log_text[-CANDIDATE_OUTPUT_BYTES:],
        "error": None if exit_code == 0 else f"the candidate exited {exit_code} on Kaggle",
        "hours": round(hours or 0.0, 4),
        # Kaggle rations hours, not dollars -- see `core/kaggle_quota.py`. The
        # zero is a fact about the backend rather than a missing measurement, and
        # `hours` is the number that bounds a campaign here.
        "cost_usd": 0.0,
        "accelerator": accelerator,
        "accelerator_kind": kind,
        "kernel_status": state.get("status"),
        "where": _url(ref),
    }


def _queue_grace(cfg: Config) -> float:
    """How long a smoke run may sit in Kaggle's queue before the poll gives up.

    Its own setting rather than a multiple of the status timeout: they answer
    unrelated questions -- one is how long a single API call may take, the other
    is how busy Kaggle is -- and deriving the second from the first means tuning
    either one silently moves the other.
    """
    return float(cfg.get("kaggle", "queue_grace_s", 900))


def _wait(cfg: Config, ref: str, *, deadline: float) -> dict[str, Any]:
    interval = float(cfg.get("kaggle", "poll_interval_s", 30))
    state = _status(cfg, ref)
    while state.get("status") not in _TERMINAL and time.time() < deadline:
        time.sleep(interval)
        state = _status(cfg, ref)
    return state


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def _log_hours(artifacts: Path, slug: str) -> tuple[float | None, str]:
    """Execution hours from the kernel log, and the log as text.

    Kaggle's log is a JSON array of `{stream_name, time, data}` where `time` is
    seconds since the kernel started, so the largest one is how long the kernel
    actually ran. That is a better number than wall clock since submit, which
    includes however long Kaggle kept the kernel queued -- charging queue time to
    a weekly allowance would make the allowance shrink under load rather than
    under use. Returns None when the log is absent or in a shape this does not
    recognise, and the caller says so rather than pretending to precision.
    """
    candidates = [artifacts / f"{slug}.log", *sorted(artifacts.glob("*.log"))]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, list):
            continue
        times = [
            float(rec["time"])
            for rec in doc
            if isinstance(rec, dict) and isinstance(rec.get("time"), (int, float))
        ]
        lines = "\n".join(
            str(rec.get("data", "")).rstrip("\n")
            for rec in doc
            if isinstance(rec, dict)
        )
        if times:
            return max(times) / 3600.0, lines
        return None, lines
    return None, ""


def _fetch_output(cfg: Config, ref: str, artifacts: Path) -> tuple[dict[str, Any], float | None, str]:
    """Download everything the kernel produced, then read the marker and the log."""
    slug = ref.split("/", 1)[-1]
    try:
        _run(
            [_executable(), "kernels", "output", ref, "-p", str(artifacts), "--force"],
            cfg,
            timeout=float(cfg.get("kaggle", "output_timeout_s", 1800)),
        )
    except GradError:
        # A kernel that errored early may have produced no downloadable output at
        # all. That is a collectable outcome, not a reason to leave the run
        # in-flight for the stale gate to trip over later.
        pass
    marker: dict[str, Any] = {}
    marker_path = artifacts / MARKER
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            marker = {}
    hours, log_text = _log_hours(artifacts, slug)
    if hours is None and isinstance(marker.get("elapsed_s"), (int, float)):
        # The notebook timed its own subprocess, which is the tightest number
        # available: it excludes both the queue and the unpack cell.
        hours = float(marker["elapsed_s"]) / 3600.0
    return marker, hours, log_text


def _collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")
    p.add_argument("--wait", action="store_true", help="poll until the kernel finishes")
    p.add_argument("--timeout", type=int, default=900, help="seconds, with --wait")
    p.add_argument(
        "--delete-kernel",
        action="store_true",
        help="delete the kernel from Kaggle after collecting (off by default; the kernel is the run's public record)",
    )


@cli.command("collect", "fetch artifacts, compute deviations, write the run record", setup=_collect_args)
def cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    r = submit_lib.require_uncollected(args.run_id)
    handle = r.get("handle") or {}
    ref = handle.get("kernel_ref")
    if not ref:
        raise GradError(
            "no_handle",
            f"run {r.id} has no kernel reference; it never reached Kaggle",
            exit_code=3,
            fix=f"python -m tools.ledger show {r.id} --json",
        )

    # `_wait` polls once before sleeping, so the non-waiting case is the same
    # call with a deadline already past rather than a second code path.
    state = _wait(cfg, ref, deadline=time.time() + (args.timeout if args.wait else 0))
    if state.get("status") not in _TERMINAL:
        raise GradError(
            "still_running",
            f"kernel {ref} is {state.get('status')}",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.kaggle collect {r.id} --wait --timeout 3600 --json",
            detail={"run_id": r.id, "kernel": state},
        )

    artifacts = submit_lib.artifacts_dir(r.id)
    marker, log_hours, log_text = _fetch_output(cfg, ref, artifacts)
    if log_text:
        (artifacts / "kernel.log").write_text(log_text, encoding="utf-8")

    results: dict[str, Any] = {}
    # Every value the run reported per quantity, not just the last. A run that
    # reports one quantity several times is a replicated run -- see core/stats.py.
    samples: dict[str, list[Any]] = {}
    metrics_error = None
    try:
        results, samples = submit_lib.read_metrics(
            artifacts / Path(r.get("metrics_file") or "metrics.json").name
        )
    except GradError as exc:
        metrics_error = exc.message

    expectation = None
    if r.get("expectation_id"):
        try:
            expectation = ls.expectation(r["expectation_id"])
        except GradError:
            expectation = None

    if log_hours is not None:
        hours, basis = log_hours, "kernel log (execution time, excludes queueing)"
    else:
        hours = submit_lib.elapsed_hours(r)
        basis = "wall clock since submit -- the kernel log was unreadable, so this overcounts any time Kaggle spent queueing"

    exit_code = marker.get("exit_code")
    # Kaggle's own verdict is the outer one and the marker is the inner one, and
    # a run needs both to be good. A kernel stopped at the session cap reports
    # `error` with no marker at all; one whose entrypoint failed reports `error`
    # with an exit code. Trusting only the marker would call the first a success
    # because nothing said otherwise.
    ok = state.get("status") == "complete" and exit_code in (0, None)
    record = submit_lib.finish(
        r.id,
        status="completed" if ok else "failed",
        # Free, and stated as a number so the ledger folds uniformly.
        cost_usd_actual=0.0,
        results=results,
        artifacts_dir=artifacts,
        expectation=expectation,
        samples=samples,
        extra={
            "exit_code": exit_code,
            "kernel_ref": ref,
            "kernel_status": state.get("status"),
            "kernel_message": state.get("message"),
            kaggle_quota.F_ACTUAL: round(hours, 4),
            "accelerator_hours_basis": basis,
            "cost_basis": "Kaggle kernels are free; the metered resource is accelerator hours, not dollars",
            "metrics_error": metrics_error,
        },
    )

    if args.delete_kernel:
        try:
            _run([_executable(), "kernels", "delete", ref, "-y"], cfg,
                 timeout=float(cfg.get("kaggle", "status_timeout_s", 120)))
        except GradError:
            pass

    unjudged = [d for d in record["deviations"] if d.get("in_range") is not True]
    return {
        "run": record,
        "artifacts": str(artifacts),
        "accelerator_hours": round(hours, 4),
        "quota": kaggle_quota.summary(cfg),
        "needs_verdict": unjudged,
        "next": (
            f"python -m tools.ledger verdict {r.id} --quantity {unjudged[0]['quantity']} "
            "--verdict bug|real|inconclusive --note '...' --json"
        ) if unjudged else None,
    }


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
def _account_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="username", metavar="USERNAME", help="the Kaggle account to run as")
    p.add_argument("--clear", action="store_true", help="forget the stored account")
    p.add_argument(
        "--check",
        action="store_true",
        help="ask Kaggle whether the account and the stored key actually authenticate",
    )


@cli.command("account", "show, set, or verify which Kaggle account runs the kernels", setup=_account_args)
def cmd_account(args: argparse.Namespace) -> dict[str, Any]:
    """Which account, and whether it works.

    The username is not a credential and is deliberately not stored as one: only
    the key is secret, and an account name you cannot read back is a worse
    answer to "whose kernels are these?" than a file you can. The key still goes
    to Windows Credential Manager -- `python -m tools.jobs credential set
    kaggle_key` -- and the two halves are useless apart, which is what `--check`
    is for.
    """
    from core import jsonl  # noqa: PLC0415

    cfg = config_mod.load()
    if args.username and args.clear:
        raise UsageError("--set and --clear ask for opposite things", fix="pass one of them")

    shadowed = None
    if args.clear:
        account_path().unlink(missing_ok=True)
    elif args.username:
        name = validate_username(args.username)
        jsonl.write_json(account_path(), {"username": name, "set_at": ls.now_iso()})
        configured = str(cfg.get("kaggle", "username", "") or "").strip()
        if configured and configured != name:
            # Said out loud rather than left to be discovered: the file now wins,
            # and a config value that no longer takes effect is exactly the kind
            # of thing someone spends an afternoon on later.
            shadowed = configured

    name, source = resolve_username(cfg)
    payload: dict[str, Any] = {
        "username": name,
        "source": source,
        "path": str(account_path()),
        "key_stored": credentials.present(credentials.KAGGLE_KEY),
        "shadowed_config_username": shadowed,
    }
    if not name:
        payload["fix"] = "python -m tools.kaggle account --set <your kaggle username> --json"
    elif not payload["key_stored"]:
        payload["fix"] = f"python -m tools.jobs credential set {credentials.KAGGLE_KEY}"

    if args.check:
        payload["check"] = _check_account(cfg)
    return payload


def _check_account(cfg: Config) -> dict[str, Any]:
    """One cheap authenticated call, so "is this set up right?" has an answer.

    `kernels list --mine` is the smallest thing that proves both halves: a wrong
    username and a wrong key fail it the same way, which is fine -- they have the
    same fix, which is to look at the pair.
    """
    try:
        output = _run(
            [_executable(), "kernels", "list", "--mine", "--page-size", "1"],
            cfg,
            timeout=float(cfg.get("kaggle", "status_timeout_s", 120)),
        )
    except GradError as exc:
        return {
            "ok": False,
            "reason": exc.message,
            "fix": (
                exc.fix
                or "check the username against your Kaggle profile URL and re-set the key: "
                f"python -m tools.jobs credential set {credentials.KAGGLE_KEY}"
            ),
        }
    return {"ok": True, "output": "\n".join((output or "").splitlines()[:5])}


@cli.command("quota", "how many accelerator hours the rolling window has left")
def cmd_quota(_: argparse.Namespace) -> dict[str, Any]:
    return kaggle_quota.summary(config_mod.load())


@cli.command("accelerators", "the accelerator inventory and which pool each draws from")
def cmd_accelerators(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    table = cfg.get("kaggle", "accelerators", {}) or {}
    return {
        "default": cfg.get("kaggle", "default_accelerator"),
        "accelerators": [
            {"id": name, "pool": kind, "session_cap_hours": kaggle_quota.session_cap(cfg, kind)}
            for name, kind in sorted(table.items(), key=lambda kv: (kv[1], kv[0]))
        ],
        "note": (
            "an id absent from this table is a configuration error, never a guess: charging a "
            "TPU run to the GPU pool and charging it to no pool are both ways for the weekly "
            "ceiling to stop bounding what it exists to bound"
        ),
    }


if __name__ == "__main__":
    main(cli)
