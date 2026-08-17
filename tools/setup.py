"""grad-setup -- the answers a wizard is allowed to write.

Three questions, and none of them belong in `config/grad.toml`: which model runs
which role, which backend to reach for by default, and which SSH hosts exist.
That file is hand-annotated and `tomllib` cannot write it, so a command that
edited it would reformat it and drop every comment in it. The answers go to
`core/settings.py` instead -- an overlay under the app directory, keyed by
workspace, which outranks the file and *says so*.

`show` is the important command here. A layered resolution that cannot be
inspected is a layered resolution that will eventually be argued with, and the
argument is always the same one: someone edits `[models] evolve`, sees no
change, and has no way to find out that a file they have never heard of wins.

Every command here is what the setup window's buttons run (§10), which is what
keeps the window from growing a second way to do any of it.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import config as config_mod, credentials, settings
from core.cli import Cli, main
from core.errors import UsageError

cli = Cli(
    "grad-setup",
    "Models, backends and SSH hosts: the writable half of the configuration.",
    epilog=(
        "These are stored under the app directory, per workspace, and they win over\n"
        "config/grad.toml -- which is hand-annotated and cannot be machine-written\n"
        "without losing every comment in it. `show` reports what is overriding what.\n\n"
        "Credentials are not here: they live in the OS credential store\n"
        "(`python -m tools.jobs credential set <name>`). Ceilings are not here either:\n"
        "a ceiling that moved is an event, and it belongs in `python -m tools.budget raise`."
    ),
)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------
@cli.command("show", "every layer, and which one wins")
def cmd_show(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load(reload=True)
    overlay_models = settings.models()
    project_models = (cfg.project_overlay.get("models") or {})
    roles = []
    for role in config_mod.MODEL_ROLES:
        configured = (cfg.user.get("models") or {}).get(role)
        legacy = config_mod.LEGACY_MODEL_KEYS.get(role)
        from_legacy = (cfg.user.get(legacy[0]) or {}).get(legacy[1]) if legacy else None
        roles.append(
            {
                "role": role,
                "model": cfg.model_for(role),
                # The layer that actually selected it, in `model_for`'s order.
                # This listed three of the five and got the answer wrong for the
                # other two -- a role set by the selected project reported as
                # "config", and one coming from a legacy `[agent] model` key
                # reported as "default". The whole point of `show` is that the
                # resolution can be inspected, so a source that is nearly right
                # is worse here than in most places.
                "source": (
                    "project"
                    if role in project_models
                    else "setup"
                    if role in overlay_models
                    else "config"
                    if configured
                    else "legacy"
                    if from_legacy
                    else "default"
                ),
                "project": project_models.get(role),
                "overlay": overlay_models.get(role),
                "config": configured,
                "legacy": from_legacy,
                "default": config_mod.DEFAULTS["models"][role],
            }
        )

    inventory = []
    configured_hosts = cfg.raw.get("hosts") or {}
    added = settings.hosts()
    for name, host in sorted(cfg.hosts.items()):
        inventory.append(
            {
                "name": name,
                "hostname": host.hostname,
                "user": host.user,
                "rate_usd_per_hour": host.rate_usd_per_hour,
                "gpus": host.gpus,
                "source": "setup" if name in added else "config",
            }
        )

    return {
        "settings_path": str(settings.path()),
        "config_path": str(config_mod.paths.config_path()),
        "models": roles,
        "backend": {"default": settings.default_backend(), "known": list(settings.BACKENDS)},
        "hosts": inventory,
        # The report that makes winning acceptable. `kaggle account` says the
        # same thing for the same reason.
        "shadowing": settings.shadowing(cfg),
        "config_hosts": sorted(configured_hosts),
    }


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _models_args(p: argparse.ArgumentParser) -> None:
    # One flag per role, derived from the roles rather than written out, so a
    # seventh role is reachable here the moment it exists.
    for role in config_mod.MODEL_ROLES:
        p.add_argument(
            f"--{role}",
            metavar="MODEL",
            help=f"the model for the {role} role (default: {config_mod.DEFAULTS['models'][role]})",
        )
    p.add_argument(
        "--clear",
        action="append",
        default=[],
        metavar="ROLE",
        help="drop an override, so the role falls back to the config and then the default",
    )


@cli.command("models", "choose the model for one or more roles", setup=_models_args)
def cmd_models(args: argparse.Namespace) -> dict[str, Any]:
    """Six roles, and the two the funnel does not name here.

    `[retrieval] rerank_model` and `embed_model` are deliberately not settable
    through this command. They are a different provider on a different billing
    rail -- Voyage costs credits, the roles below cost subscription quota -- and
    `config/grad.toml` argues at length against folding the two together. A
    wizard that offered all eight in one list would be making exactly the
    substitution §16 exists to prevent.
    """
    chosen = {
        role: getattr(args, role) for role in config_mod.MODEL_ROLES if getattr(args, role, None)
    }
    if not chosen and not args.clear:
        raise UsageError(
            "nothing to set",
            fix=f"--{config_mod.MODEL_ROLES[0]} claude-opus-5   # or --clear {config_mod.MODEL_ROLES[0]}",
        )
    if chosen:
        settings.set_models(chosen)
    if args.clear:
        settings.clear_models(list(args.clear))
    cfg = config_mod.load(reload=True)
    return {
        "set": chosen,
        "cleared": list(args.clear),
        "models": cfg.models(),
        "shadowing": settings.shadowing(cfg),
    }


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
def _backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--default",
        dest="default_backend",
        choices=settings.BACKENDS,
        help="which backend to reach for when nothing more specific applies",
    )


@cli.command("backend", "choose the default backend", setup=_backend_args)
def cmd_backend(args: argparse.Namespace) -> dict[str, Any]:
    """A default, not a restriction.

    The three backends are not alternatives and the useful setup is a mixture:
    Kaggle's free hours for a smoke run, HF Jobs for the one that matters. So
    this records a preference and refuses nothing -- `--remote` still names a
    backend per campaign, and a spec's `[target]` still wins over both.
    """
    if args.default_backend:
        settings.set_backend(args.default_backend)
    return {
        "default": settings.default_backend(),
        "known": list(settings.BACKENDS),
        "readiness": readiness(config_mod.load(reload=True)),
    }


# ---------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------
def _host_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=("add", "remove"))
    p.add_argument("--name", required=True, help="what Grad calls this host")
    p.add_argument("--hostname", help="what ssh connects to")
    p.add_argument("--user", default="", help="the ssh user")
    p.add_argument(
        "--rate",
        dest="rate_usd_per_hour",
        default=0.0,
        help="dollars per hour, for pricing wall clock at collect time. 0 for a free host",
    )
    p.add_argument("--workdir", default="~/grad", help="where runs are staged on the host")
    p.add_argument("--gpus", default=1, type=int)
    p.add_argument(
        "--key-credential",
        dest="key_credential",
        help="the keyring entry holding this host's key. never a path to a key file",
    )
    p.add_argument("--notes", default="", help="anything worth remembering about this box")


@cli.command("host", "add or remove an SSH host in the inventory", setup=_host_args)
def cmd_host(args: argparse.Namespace) -> dict[str, Any]:
    """The inventory stays fixed; this gives it a second, writable source.

    `core/config.py:host` refuses an unknown name because a host that can be
    named ad-hoc is a general remote-execution capability the threat model does
    not grant. Nothing about that changes here -- a host has to be added, on
    purpose, before anything can reach it, and the refusal now names both places
    it could have been added.
    """
    if args.action == "remove":
        settings.remove_host(args.name)
    else:
        settings.add_host(
            args.name,
            {
                "hostname": args.hostname,
                "user": args.user,
                "rate_usd_per_hour": args.rate_usd_per_hour,
                "workdir": args.workdir,
                "gpus": args.gpus,
                "key_credential": args.key_credential,
                "notes": args.notes,
            },
        )
    cfg = config_mod.load(reload=True)
    return {
        "action": args.action,
        "name": args.name,
        "hosts": sorted(cfg.hosts),
        "added_here": sorted(settings.hosts()),
    }


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
#: What each backend needs before it can be submitted to, and what to run when it
#: is missing. `hf_token` is *required for HF Jobs* -- which is a different claim
#: from "required", and the credentials panel used to make the stronger one at a
#: user who had chosen Kaggle.
REQUIREMENTS: dict[str, dict[str, Any]] = {
    "ssh": {
        "credentials": (),
        "needs_host": True,
        "fix": "python -m tools.setup host add --name gpu-box --hostname … --user … --json",
    },
    "hf_jobs": {
        "credentials": (credentials.HF_TOKEN,),
        "needs_host": False,
        "fix": "python -m tools.jobs credential set hf_token",
    },
    "kaggle": {
        "credentials": (credentials.KAGGLE_KEY,),
        "needs_host": False,
        "needs_kaggle_account": True,
        "fix": "python -m tools.kaggle account --set <username> --json",
    },
}


def readiness(cfg: config_mod.Config) -> list[dict[str, Any]]:
    """Which backends could actually take a submission right now.

    Reads state; runs nothing. `kaggle account --check` makes a real
    authenticated call and this deliberately does not -- a readiness report that
    takes several seconds and touches the network is one nothing will call on a
    window's refresh.
    """
    from tools import kaggle as kaggle_tool

    stored = credentials.status()
    out = []
    for backend, needs in REQUIREMENTS.items():
        missing = [name for name in needs["credentials"] if not stored.get(name)]
        if needs.get("needs_host") and not cfg.hosts:
            missing.append("an ssh host")
        if needs.get("needs_kaggle_account"):
            username, _ = kaggle_tool.resolve_username(cfg)
            if not username:
                missing.append("a kaggle username")
        out.append(
            {
                "backend": backend,
                "ready": not missing,
                "missing": missing,
                "fix": None if not missing else needs["fix"],
            }
        )
    return out


@cli.command("check", "what is still missing, per backend")
def cmd_check(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load(reload=True)
    stored = credentials.status()
    backend_readiness = readiness(cfg)
    return {
        # The subscription token is not a backend's business: it is what runs the
        # agent at all, so it is reported on its own rather than folded into a
        # row about GPUs.
        "agent": {
            "ready": bool(stored.get(credentials.CLAUDE_TOKEN)),
            "fix": (
                None
                if stored.get(credentials.CLAUDE_TOKEN)
                else "claude setup-token, then: python -m tools.jobs credential set claude_oauth_token"
            ),
        },
        "backends": backend_readiness,
        "any_backend_ready": any(r["ready"] for r in backend_readiness),
        "default_backend": settings.default_backend(),
        "models": cfg.models(),
    }


if __name__ == "__main__":
    main(cli)
