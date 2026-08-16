#!/usr/bin/env bash
# Install Grad: a virtual environment, the dependencies, and a `grad` launcher.
#
# This is the portable half of the installer. It covers three situations, and it
# is worth being clear about which one you are in, because they do not all get
# the same app:
#
#   * Git Bash / MSYS on Windows -- the full desktop app. Identical result to
#     install.ps1, except that the Start Menu shortcut is that script's job:
#     creating a .lnk needs COM. Run install.ps1 if you want the shortcut.
#   * WSL -- the CLI and the browser UI. The native window needs WebView2, which
#     is on the Windows side of the boundary, so `--ui` falls back to a browser.
#   * Linux / macOS -- the CLI and the browser UI. The workspace, the ledger,
#     the gates and the notebooks all work; the native window and the
#     notification-area icon are Windows features and are skipped.
#
# What no installer on any platform can do is vendor the `claude` CLI. The SDK
# spawns it as a subprocess and speaks stream-JSON over its stdio, so it is a
# separate native binary with its own auth. This script checks for it and tells
# you how to get it.

set -euo pipefail

step() { printf '\033[36m==> %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m+\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '  \033[31mx\033[0m %s\n' "$1" >&2; exit 1; }

# `curl .../install.sh | bash` runs this text with no file behind it, so
# BASH_SOURCE is empty (and under `set -u`, asking for it unguarded would kill
# the script). No file -- or a copy saved alone, with no pyproject.toml beside
# it -- means there is no repository to install from, and the repository *is*
# the install: prompts, skills and workspace are read from it at runtime. So
# clone it, then run the clone's own copy of this script.
SELF="${BASH_SOURCE[0]:-}"
if [ -n "$SELF" ] && [ -f "$(dirname "$SELF")/pyproject.toml" ]; then
  ROOT="$(cd "$(dirname "$SELF")" && pwd)"
else
  REPO_URL="https://github.com/view321/Grad.git"
  step "No repository behind this script; cloning Grad"
  command -v git >/dev/null 2>&1 || die "git is not installed; install it and re-run"
  DEST="${XDG_DATA_HOME:-$HOME/.local/share}/grad/app"
  if [ -f "$DEST/pyproject.toml" ]; then
    ok "reusing the existing clone at $DEST"
    git -C "$DEST" pull --ff-only >/dev/null 2>&1 \
      || warn "could not fast-forward it; installing what is there"
  else
    mkdir -p "$(dirname "$DEST")"
    git clone "$REPO_URL" "$DEST"
    ok "cloned into $DEST"
  fi
  exec bash "$DEST/install.sh"
fi
VENV="$ROOT/.venv"
# The set the app needs at runtime, which is not the set it needs to start.
# `remote` is keyring -- every credential path -- and storing a credential is
# the step immediately after installing. `retrieval` is httpx and sqlite-vec, so
# without it the paper funnel cannot run. `math` is the SymPy the system prompt
# tells the agent to reach for by name. None of the three fails at install time;
# they fail later, one feature at a time, on a machine where the developer
# already had all of them.
EXTRAS="${GRAD_EXTRAS:-ui,notebook,agent,lab,retrieval,remote,math}"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
  Darwin)               PLATFORM="macos" ;;
  *)                    PLATFORM="linux" ;;
esac
if [ "$PLATFORM" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
  PLATFORM="wsl"
fi

# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
step "Locating Python 3.11 or newer"
PYTHON=""
for candidate in "${GRAD_PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
  [ -n "$candidate" ] || continue
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
[ -n "$PYTHON" ] || die "No Python 3.11+ found. Install one and re-run, or set GRAD_PYTHON."
ok "$($PYTHON -c 'import sys; print("Python %d.%d at %s" % (*sys.version_info[:2], sys.executable))')"

# ---------------------------------------------------------------------------
# 2. The environment
# ---------------------------------------------------------------------------
step "Creating the virtual environment"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
  ok "created $VENV"
else
  ok "reusing $VENV"
fi

# Windows venvs put the interpreter in Scripts/, POSIX in bin/. Git Bash sees
# the Windows layout, so this cannot key off the shell.
if [ -x "$VENV/Scripts/python.exe" ]; then
  VPY="$VENV/Scripts/python.exe"
  VPYW="$VENV/Scripts/pythonw.exe"
else
  VPY="$VENV/bin/python"
  VPYW=""
fi
[ -x "$VPY" ] || die "The virtual environment has no interpreter at $VPY"

step "Installing Grad and its dependencies (this takes a few minutes)"
"$VPY" -m pip install --upgrade pip --quiet
# Editable, because this repository *is* the install: the workspace, the prompts
# and the skills are read from here at runtime.
"$VPY" -m pip install -e "$ROOT[$EXTRAS]" || die "pip install failed."
ok "installed extras: $EXTRAS"

# ---------------------------------------------------------------------------
# 3. What cannot be vendored
# ---------------------------------------------------------------------------
step "Checking what has to be present but cannot be vendored"
if command -v claude >/dev/null 2>&1; then
  ok "claude CLI at $(command -v claude)"
else
  warn "The 'claude' CLI was not found on PATH."
  warn "The agent spawns it as a subprocess -- without it the workspace opens"
  warn "but no turn can run. Install it and re-run:"
  warn "    npm install -g @anthropic-ai/claude-code"
fi

case "$PLATFORM" in
  windows) ok "native desktop window and notification-area icon available" ;;
  wsl)
    warn "WSL: the native window needs WebView2 on the Windows side, so --ui"
    warn "runs in browser mode. Open the port it prints in a Windows browser." ;;
  *)
    warn "$PLATFORM: the native window and tray icon are Windows features."
    warn "--ui runs in browser mode; everything else is unaffected." ;;
esac

# ---------------------------------------------------------------------------
# 3b. Where the research goes
# ---------------------------------------------------------------------------
# The workspace -- ledger, notebooks, notes, figures, reports -- defaults to this
# folder, which is also the checkout. That is the simplest thing that works and
# it is what makes updating awkward: research committed into the same repository
# as the code puts your notebooks on the same branch as upstream's releases, and
# every `grad update` becomes a merge.
#
# So this asks. Keeping them apart costs nothing and means an update is a
# fast-forward over files nobody has edited.
step "Choosing where your research will live"
WORKSPACE="${GRAD_WORKSPACE:-}"
if [ -z "$WORKSPACE" ]; then
  if [ -s "$ROOT/ledger/runs.jsonl" ]; then
    # Research is already here. Moving it is `grad workspace move`, which copies
    # and verifies before it deletes anything -- not something to do silently
    # from an installer.
    WORKSPACE="$ROOT"
    warn "this folder already holds a ledger, so it stays the workspace"
    warn "to separate them later:  python -m tools.workspace move ~/Grad"
  elif [ -t 0 ]; then
    printf '  Workspace folder [%s]: ' "$HOME/Grad"
    read -r REPLY || REPLY=""
    WORKSPACE="${REPLY:-$HOME/Grad}"
  else
    WORKSPACE="$HOME/Grad"
  fi
fi

if [ "$WORKSPACE" = "$ROOT" ]; then
  ok "workspace: $ROOT (inside the installation)"
else
  "$VPY" -m tools.workspace use "$WORKSPACE" --create >/dev/null \
    && ok "workspace: $WORKSPACE" \
    || die "could not use $WORKSPACE as the workspace"
fi

# What the extras were, so `grad update` can reinstall with the same set rather
# than guessing. Dropping `ui` on an update would take the desktop app off a
# machine whose owner asked only for an update.
"$VPY" - "$EXTRAS" <<'PY' >/dev/null 2>&1 || true
import sys
from core import update
update.write_install_record(
    extras=[x.strip() for x in sys.argv[1].split(",") if x.strip()],
    installer="install.sh",
)
PY

# ---------------------------------------------------------------------------
# 4. The launcher
# ---------------------------------------------------------------------------
step "Creating the launcher"
BIN_DIR="${GRAD_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN_DIR"
LAUNCHER="$BIN_DIR/grad"

# pythonw.exe where there is one: on Windows the console interpreter would keep
# a black window open behind the app for its whole lifetime, which is the same
# problem core/spawn.py solves for every child process.
if [ -n "$VPYW" ] && [ -x "$VPYW" ]; then
  RUNNER="$VPYW"
else
  RUNNER="$VPY"
fi

cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Generated by install.sh. Re-run it after moving the workspace.
# No --port: the app takes 8080, or the next free port above it. That is
# deliberate rather than random -- JupyterLab fixes the origins it will be
# framed by at launch, so a port that moves every time means a Lab server that
# has to be restarted every time. See ui/desktop.py:choose_port.
exec "$RUNNER" "$ROOT/agent.py" "\$@"
EOF
chmod +x "$LAUNCHER"
ok "$LAUNCHER"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on your PATH. Add it:"
     warn "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc" ;;
esac

# ---------------------------------------------------------------------------
# 5. Where things live
# ---------------------------------------------------------------------------
if [ "$PLATFORM" = "windows" ]; then
  APP_STATE="%LOCALAPPDATA%\\Grad"
else
  APP_STATE="$HOME/.local/state/grad"
fi

printf '\n'
step "Done"
printf '  Installation (the code, replaced by updates):  %s\n' "$ROOT"
printf '  Workspace (your research, never touched):     %s\n' "$WORKSPACE"
printf '  App state (layouts, logs, Lab token):         %s\n' "$APP_STATE"
printf '\n'
printf '  Start the workspace:   grad --ui\n'
printf '  Ask a single question: grad "what is in the ledger?"\n'
printf '  Update to the newest:  grad --update\n'
printf '\n'
if [ "$PLATFORM" = "windows" ]; then
  printf '  For a Start Menu shortcut, run install.ps1 as well:\n'
  printf '      powershell -ExecutionPolicy Bypass -File %s\n\n' "$ROOT/install.ps1"
fi
