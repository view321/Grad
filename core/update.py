"""Updating Grad in place, from the git checkout it was installed from.

The install is a checkout and the install is editable (`pip install -e .`), and
both halves of that shape the whole design here.

**Editable means a fast-forward is usually the entire update.** The interpreter
imports the working tree, so a release that changed only Python code is live the
moment `git merge --ff-only` returns; there is nothing to build and nothing to
copy. The reinstall step is therefore *conditional*, on `pyproject.toml` having
changed between the two commits -- which is the only thing that can alter what
must be present in the environment. That is the difference between a two-second
update and a two-minute one, on the overwhelmingly common release.

**Releases, not `main`.** The target is the newest `v*` tag, so a user moves
between named versions they can cite in a paper, `--to` can pin an older one,
and `--rollback` is free. "Reproduce last month's result" means "run last
month's code", and for a tool whose output is meant to be checkable that is not
a nicety.

**What may block it, and what may not.** A modified *code* file blocks: the
merge would fail anyway, and failing early names the file. Research does not
block, even though on the default layout it sits in the same checkout --
`core/version.py:WORKSPACE_PATHS` is the split, and it is only a real conflict
when the incoming commits touch a path the user has also modified, which is
checked exactly rather than assumed. A running instance blocks only the
*reinstall*: swapping files under a live process is safe for an editable
checkout, since imported modules are already in memory, but replacing a
dependency in the environment a lazy `import nicegui` is about to reach is not.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from importlib import util as import_util
from pathlib import Path
from typing import Any

from core import appdata, migrate, spawn, version
from core.errors import EXIT_CONFIG, EXIT_UPSTREAM, GradError, UsageError

log = logging.getLogger("grad.update")

#: What the installers install when nobody says otherwise. Kept in step with
#: `install.sh` and `install.ps1`; it is the fallback for a checkout whose
#: `install.json` is missing and whose extras could not be detected.
DEFAULT_EXTRAS = ("ui", "notebook", "agent", "lab")

#: extra -> a module that is present if and only if it was installed. Used to
#: reconstruct the extras of an installation that predates `install.json`
#: (`core/migrate.py`, step 2). `find_spec` rather than `import`, because
#: importing NiceGUI to find out whether NiceGUI is installed has side effects
#: the suite explicitly forbids.
EXTRA_PROBES: dict[str, str] = {
    "agent": "claude_agent_sdk",
    "notebook": "jupyter_client",
    "retrieval": "sqlite_vec",
    "remote": "huggingface_hub",
    "ui": "nicegui",
    "lab": "jupyterlab",
    "lab-extensions": "jupyterlab_lsp",
    "math": "sympy",
    "wiki": "repowiki",
    "evolve": "shinka",
    "dev": "pytest",
}

#: A fetch crosses the network; everything else in `core/version.py` does not,
#: and the default timeout there is sized for a local call.
FETCH_TIMEOUT_S = 60.0
#: A reinstall resolves and downloads wheels. `python-lsp-server[all]` is the
#: one that makes this a real number rather than a formality.
PIP_TIMEOUT_S = 900.0
#: How stale the cached answer may be before the app checks again. Once a day:
#: often enough to notice a release, rarely enough that it is not a background
#: `git fetch` every time someone opens the window.
CHECK_INTERVAL_S = 86400.0


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# what this installation is, and what it was installed with
# ---------------------------------------------------------------------------
def install_record_path() -> Path:
    """Beside the other per-installation state, not in the checkout.

    In the checkout it would be an untracked file that every `git status` in the
    repository reports, and -- worse -- one more thing to explain to a user who
    is being told their tree must be clean before it can be updated.
    """
    return appdata.state_dir() / "install.json"


def read_install_record() -> dict[str, Any]:
    try:
        data = json.loads(install_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_install_record(*, extras: list[str] | tuple[str, ...], **fields: Any) -> None:
    """Record what this installation was installed with. Never raises."""
    appdata.ensure()
    payload = {**read_install_record(), "extras": list(extras), **fields}
    try:
        install_record_path().write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        log.debug("could not write %s", install_record_path())


def begin_update(target: dict[str, Any], *, needs_reinstall: bool, chosen: list[str]) -> None:
    """Record an update that is about to start, before anything moves.

    An update is three steps -- move the checkout, reinstall, migrate -- and
    only the first is atomic. Without this marker a pip failure in step 2 is
    unrecoverable *by the updater*: the commit has already advanced, so the next
    `plan()` computes `available = head != target` as False, `apply` answers
    "already up to date", and the environment is left running new code against
    old dependencies with no path back. The migrations, which run after the
    reinstall, are skipped for the same reason and never revisited.

    Version strings cannot substitute for this. With an editable install the
    metadata is written at `pip install` time, so after any successful update
    that *correctly* skipped the reinstall it is permanently behind
    `pyproject.toml` -- a mismatch that would then demand a reinstall forever.
    What is needed is not "do the versions agree" but "did the step I started
    finish", and that is a fact only the updater can record.
    """
    write_install_record(
        extras=chosen,
        incomplete={
            "tag": target.get("tag"),
            "commit": target.get("commit"),
            "needs_reinstall": bool(needs_reinstall),
            "extras": list(chosen),
            "started_at": now_iso(),
        },
    )


def finish_update() -> None:
    """Clear the marker. Called only once every step has actually run."""
    record = read_install_record()
    record.pop("incomplete", None)
    appdata.ensure()
    try:
        install_record_path().write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        # The update completed; only the bookkeeping failed. The cost is one
        # unnecessary reinstall on the next `grad update`, which is idempotent.
        log.debug("could not clear the incomplete-update marker")


def incomplete_update() -> dict[str, Any] | None:
    """The update that started and did not finish, if there is one."""
    value = read_install_record().get("incomplete")
    return value if isinstance(value, dict) else None


def detect_extras() -> list[str]:
    """Which optional dependency sets are actually present."""
    found: list[str] = []
    for extra, module in EXTRA_PROBES.items():
        try:
            if import_util.find_spec(module) is not None:
                found.append(extra)
        except (ImportError, ValueError):
            continue
    return found


def extras() -> list[str]:
    """The extras a reinstall should use.

    Recorded first, detected second, defaulted last. Getting this wrong is not
    cosmetic: reinstalling without `ui` would take the desktop app off the
    machine of someone who asked only for an update.
    """
    recorded = read_install_record().get("extras")
    if isinstance(recorded, list) and recorded:
        return [str(x) for x in recorded]
    return detect_extras() or list(DEFAULT_EXTRAS)


# ---------------------------------------------------------------------------
# git questions
# ---------------------------------------------------------------------------
def remote() -> str | None:
    """The remote to fetch from: `origin` if there is one, else the first."""
    listed = version.git("remote")
    names = [line.strip() for line in (listed or "").splitlines() if line.strip()]
    if not names:
        return None
    return "origin" if "origin" in names else names[0]


def release_tags() -> list[str]:
    """Release tags, newest first.

    `--sort=-v:refname` sorts by version rather than lexically, which is the
    difference between v0.10.0 following v0.9.0 and preceding it.
    """
    listed = version.git("tag", "--list", "v*", "--sort=-v:refname")
    return [line.strip() for line in (listed or "").splitlines() if line.strip()]


def _commit_of(ref: str) -> str | None:
    return version.git("rev-parse", f"{ref}^{{commit}}")


def _is_ancestor(a: str, b: str) -> bool:
    """Whether `a` is an ancestor of `b` -- i.e. whether b is a fast-forward."""
    result = version.git_result("merge-base", "--is-ancestor", a, b)
    return bool(result is not None and result.returncode == 0)


def _count_between(a: str, b: str) -> int:
    text = version.git("rev-list", "--count", f"{a}..{b}")
    try:
        return int(text or 0)
    except ValueError:
        return 0


def _changed_paths(a: str, b: str) -> list[str]:
    text = version.git("diff", "--name-only", a, b)
    return [line.strip().replace("\\", "/") for line in (text or "").splitlines() if line.strip()]


def fetch() -> str | None:
    """Update the remote refs and tags. Returns an error message, or None.

    Not fatal to a plan: an offline machine should still be able to say what it
    has and what it is, and "could not reach the remote" is a line in the
    answer rather than an exception out of it.
    """
    name = remote()
    if not name:
        return "this checkout has no git remote to fetch from"
    result = version.git_result(
        "fetch", "--tags", "--quiet", name, timeout=FETCH_TIMEOUT_S
    )
    if result is None:
        return "git could not be run"
    if result.returncode != 0:
        return (result.stderr or "git fetch failed").strip().splitlines()[-1][:300]
    return None


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def plan(*, do_fetch: bool = True, to: str | None = None) -> dict[str, Any]:
    """What an update would do, and everything standing in its way.

    Never raises for an ordinary "no" -- an installation that is not a checkout,
    a machine with no network, a tree with local edits -- because this is what
    the app's menu renders, and a menu that cannot draw is not an answer. The
    blockers are data. `apply` is the one that refuses.
    """
    identity = version.identity(reload=True)
    result: dict[str, Any] = {
        "checked_at": now_iso(),
        "current": identity,
        "label": version.label(identity),
        "target": None,
        "behind": 0,
        "available": False,
        "needs_reinstall": False,
        "blockers": [],
        "warnings": [],
        "fetch_error": None,
        "incomplete": incomplete_update(),
    }
    if result["incomplete"]:
        result["warnings"].append(
            {
                "code": "incomplete_update",
                "message": (
                    f"an update to {result['incomplete'].get('tag')} moved the checkout but did "
                    "not finish installing its dependencies"
                ),
                "fix": "grad update   # picks up where it stopped",
            }
        )

    if not version.is_checkout():
        result["blockers"].append(
            {
                "code": "not_a_checkout",
                "message": "this Grad was not installed from a git checkout",
                "fix": (
                    "clone the repository and run install.sh (or install.ps1), then "
                    "point it at your workspace with: grad workspace use <folder>"
                ),
            }
        )
        return result

    if do_fetch:
        result["fetch_error"] = fetch()

    head = _commit_of("HEAD")
    tags = release_tags()
    target_tag = to or (tags[0] if tags else None)
    if not target_tag:
        result["warnings"].append(
            {
                "code": "no_releases",
                "message": "the remote has no version tags, so there is nothing to update to",
                "fix": "tag a release upstream (v0.2.0), or track the branch manually with git",
            }
        )
        return result

    target_commit = _commit_of(target_tag)
    if not target_commit:
        result["blockers"].append(
            {
                "code": "unknown_tag",
                "message": f"no such release: {target_tag}",
                "fix": f"known releases: {', '.join(tags[:10]) or '(none)'}",
            }
        )
        return result

    forward = bool(head and _is_ancestor(head, target_commit))
    result["target"] = {
        "tag": target_tag,
        "commit": target_commit[:12],
        "direction": "forward" if forward else "checkout",
    }
    result["behind"] = _count_between(head or "HEAD", target_commit) if forward else 0
    result["available"] = bool(head and head != target_commit)
    if not result["available"]:
        return result

    changed = _changed_paths(head or "HEAD", target_commit)
    result["needs_reinstall"] = "pyproject.toml" in changed
    result["changed_files"] = len(changed)

    # -- what stands in the way ---------------------------------------------
    modified_code = version.dirty_paths()
    if modified_code:
        result["blockers"].append(
            {
                "code": "dirty_code",
                "message": (
                    f"{len(modified_code)} modified file(s) in the installation: "
                    + ", ".join(modified_code[:5])
                    + ("…" if len(modified_code) > 5 else "")
                ),
                "fix": "commit them, or discard them with: git -C <install> checkout -- .",
            }
        )

    # A research file only matters if the incoming change also touches it. On the
    # default layout the workspace *is* the checkout, so being relaxed about this
    # is the difference between an updater that works and one that is blocked
    # forever by the user's own notebooks.
    # Both halves of `status_paths` matter here: the code half blocks above, and
    # the research half is what an incoming commit could collide with. `all`
    # rather than the default, because git collapses an untracked directory into
    # one entry and a collision cannot be seen against a directory name -- see
    # `core/version.py:status_paths`.
    dirty_research = {
        p for p in version.status_paths(untracked="all") if version.is_workspace_path(p)
    }
    collisions = sorted(dirty_research.intersection(changed))
    if collisions:
        result["blockers"].append(
            {
                "code": "workspace_collision",
                "message": (
                    "the update changes files you have edited in this folder: "
                    + ", ".join(collisions[:5])
                ),
                "fix": (
                    "move your research out of the installation first: "
                    "grad workspace move <folder>"
                ),
            }
        )

    if not forward and not to:
        result["blockers"].append(
            {
                "code": "diverged",
                "message": (
                    f"{target_tag} is not ahead of this checkout; it has commits of its own"
                ),
                "fix": "reconcile the checkout with git, or pin a release: grad update --to <tag>",
            }
        )

    from core import paths  # noqa: PLC0415 - avoids a cycle through core.workspace

    if paths.root() == version.code_dir():
        result["warnings"].append(
            {
                "code": "workspace_in_install",
                "message": "your research is stored inside the installation folder",
                "fix": "grad workspace move <folder>   # keeps updates and research apart",
            }
        )
    return result


def instance_running() -> bool:
    """Whether a Grad is up, asked the only way that can be trusted.

    `core/instance.py:is_running` owns the mechanism, including why the probe
    must not go through `release()`. The window in which a launch could lose the
    race to the probe is microseconds wide and fails loudly rather than quietly.
    """
    from core import instance  # noqa: PLC0415

    try:
        return instance.is_running()
    except Exception:  # noqa: BLE001 - a guard that cannot run must not block an update
        log.debug("could not probe the instance lock", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# applying it
# ---------------------------------------------------------------------------
def apply(
    *,
    to: str | None = None,
    with_extras: str | None = None,
    force: bool = False,
    do_fetch: bool = True,
) -> dict[str, Any]:
    """Move the checkout to a release, reinstall if it has to, migrate state.

    Raises `GradError` rather than returning a refusal, because this is the half
    with side effects and a caller that ignored a returned "no" would be
    half-updated. `plan` is the one that answers without judging.
    """
    proposal = plan(do_fetch=do_fetch, to=to)
    blockers = proposal["blockers"]
    if blockers and not force:
        first = blockers[0]
        raise GradError(
            f"update_{first['code']}",
            first["message"],
            exit_code=EXIT_CONFIG,
            fix=first.get("fix"),
            detail={"blockers": blockers},
        )

    # A previous run that moved the checkout and then failed leaves work behind
    # that `available` cannot see -- the commit is already the target. Resuming
    # is the whole reason the marker exists; see `begin_update`.
    resuming = proposal["incomplete"]
    if not proposal["available"] and not resuming:
        return {
            **proposal,
            "applied": False,
            "message": f"already up to date ({proposal['label']})",
        }

    # The marker is the fallback, not just a flag: `plan` reports no target at
    # all when the remote has no release tags, and a resume must still know
    # which release it was finishing.
    target = proposal["target"] or {
        "tag": (resuming or {}).get("tag"),
        "commit": (resuming or {}).get("commit"),
        "direction": "forward",
    }
    needs_reinstall = bool(proposal["needs_reinstall"]) or bool(
        resuming and resuming.get("needs_reinstall")
    )
    if needs_reinstall and not force and instance_running():
        raise GradError(
            "update_running",
            f"{target['tag']} changes the dependencies, and Grad is running",
            exit_code=EXIT_CONFIG,
            fix="quit Grad (tray → Quit), then run: grad update",
            detail={"target": target},
        )

    before = proposal["current"].get("commit")
    chosen = [x.strip() for x in (with_extras or ",".join(extras())).split(",") if x.strip()]
    steps: list[str] = []

    # Written before anything moves, and cleared only once every step below has
    # run. A failure between the two leaves a marker the next `apply` resumes
    # from -- see `begin_update`.
    begin_update(target, needs_reinstall=needs_reinstall, chosen=chosen)

    # -- move the tree ------------------------------------------------------
    if not proposal["available"]:
        # Resuming: the checkout is already where it was asked to go, and
        # re-running the move would either be a no-op or fail on a tree that has
        # since been touched. Neither is worth risking to repeat a step that
        # demonstrably succeeded.
        steps.append(f"checkout was already at {target.get('tag')}")
    else:
        if target["direction"] == "forward":
            result = version.git_result("merge", "--ff-only", target["tag"], timeout=60.0)
        else:
            # A pin or a rollback. Detached on purpose: the checkout is *at* that
            # release rather than on a branch that happens to contain it, which is
            # the honest state and the one `git status` explains.
            result = version.git_result(
                "-c", "advice.detachedHead=false", "checkout", "--detach", target["tag"],
                timeout=60.0,
            )
        if result is None or result.returncode != 0:
            detail = (result.stderr if result else "") or "git could not be run"
            # Nothing moved, so nothing is half-done: drop the marker rather
            # than leave a resume pointing at an update that never started.
            finish_update()
            raise GradError(
                "update_failed",
                f"could not move the checkout to {target['tag']}: {detail.strip()[:300]}",
                exit_code=EXIT_UPSTREAM,
                fix="run git in the installation folder to see what it objects to",
            )
        steps.append(f"moved to {target['tag']}")

    # -- the environment, only if it has to change --------------------------
    if needs_reinstall:
        _reinstall(chosen)
        steps.append(f"reinstalled with extras: {','.join(chosen)}")
        write_install_record(extras=chosen, detected=False)
    else:
        steps.append("no dependency change; skipped the reinstall")

    # -- state --------------------------------------------------------------
    migrated = migrate.run_pending()
    if migrated:
        steps.append(f"migrated: {', '.join(migrated)}")
    finish_update()

    after = version.identity(reload=True)
    summary = {
        "applied": True,
        "from": before,
        "to": target,
        "reinstalled": needs_reinstall,
        "extras": chosen,
        "migrated": migrated,
        "steps": steps,
        "current": after,
        "label": version.label(after),
        "resumed": bool(resuming),
        "message": (
            ("finished the interrupted update to " if resuming else "updated to ")
            + str(target.get("tag"))
            + (" — restart Grad to load it" if instance_running() else "")
        ),
    }
    write_cache(plan(do_fetch=False))
    return summary


def _reinstall(chosen: list[str]) -> None:
    """`pip install -e .[extras]`, with the interpreter that is running us.

    `sys.executable` and not a `pip` on PATH: the launcher runs the app from the
    virtual environment's interpreter, and a bare `pip` could easily be another
    environment's -- which would install the new dependencies somewhere Grad
    will never import them from, and report success.
    """
    spec = str(version.code_dir()) + (f"[{','.join(chosen)}]" if chosen else "")
    argv = [sys.executable, "-m", "pip", "install", "-e", spec]
    try:
        result = spawn.run(
            argv, capture_output=True, text=True, timeout=PIP_TIMEOUT_S,
            cwd=str(version.code_dir()),
        )
    except Exception as exc:  # noqa: BLE001 - includes TimeoutExpired
        raise GradError(
            "update_reinstall_failed",
            f"the dependency install did not finish: {exc}",
            exit_code=EXIT_UPSTREAM,
            fix=f"run it yourself: {' '.join(argv)}",
        ) from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        raise GradError(
            "update_reinstall_failed",
            "pip refused the new dependencies: " + " / ".join(tail)[:400],
            exit_code=EXIT_UPSTREAM,
            fix=f"run it yourself to see the whole error: {' '.join(argv)}",
            detail={"argv": argv},
        )


def rollback_target() -> str | None:
    """The release before the one that is installed, for `--rollback`."""
    tags = release_tags()
    current_tag = version.identity().get("tag")
    if current_tag and current_tag in tags:
        index = tags.index(current_tag)
        return tags[index + 1] if index + 1 < len(tags) else None
    # Not sitting on a release: the newest tag that is already an ancestor is
    # the one this checkout came after, and going "back" means that one.
    head = _commit_of("HEAD")
    for tag in tags:
        commit = _commit_of(tag)
        if commit and head and commit != head and _is_ancestor(commit, head):
            return tag
    return None


# ---------------------------------------------------------------------------
# the cached answer the app renders
# ---------------------------------------------------------------------------
def cache_path() -> Path:
    return appdata.state_dir() / "update.json"


def write_cache(value: dict[str, Any]) -> None:
    appdata.ensure()
    try:
        cache_path().write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError:
        log.debug("could not cache the update check")


def read_cache() -> dict[str, Any]:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def cache_age_s() -> float:
    """Seconds since the last check, or `inf` if there has never been one."""
    stamp = read_cache().get("checked_at")
    if not isinstance(stamp, str):
        return float("inf")
    try:
        when = _dt.datetime.fromisoformat(stamp)
    except ValueError:
        return float("inf")
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return max(0.0, (_dt.datetime.now(_dt.timezone.utc) - when).total_seconds())


def check_due(interval_s: float = CHECK_INTERVAL_S) -> bool:
    return cache_age_s() >= interval_s


def refresh_cache() -> dict[str, Any]:
    """Check against the remote and cache the answer. For the background poll."""
    value = plan(do_fetch=True)
    write_cache(value)
    return value


def parse_extras(text: str | None) -> str | None:
    """Validate a `--extras` value before it reaches a pip command line."""
    if text is None:
        return None
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [n for n in names if n not in EXTRA_PROBES]
    if unknown:
        raise UsageError(
            f"unknown extra(s): {', '.join(unknown)}",
            fix=f"choose from: {', '.join(sorted(EXTRA_PROBES))}",
        )
    return ",".join(names)
