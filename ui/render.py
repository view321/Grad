"""A notebook you can read without opening an editor.

JupyterLab is where notebooks get *edited*, and this does not compete with that:
there is no kernel here, nothing runs, and nothing can be changed. What it
answers is the question that comes up twenty times an hour and is badly served
by a full Lab client -- *what does that cell actually do?* -- for code the agent
generated and you are reading for the first time.

Three decisions, all of which follow from "read-only" being a real constraint
rather than a description.

**`basic`, not `lab`.** nbconvert's Lab template ships a JSON payload and the
JavaScript that renders it. The `basic` template emits finished HTML with no
script tag anywhere, which is what makes the sandbox below possible -- and it
means the render survives a browser that will not run scripts from it, which is
exactly the browser we are going to give it.

**The output is untrusted.** A notebook may have arrived in a cloned repository,
and its stored outputs are arbitrary HTML that nobody re-executed. It is served
as a document to a `sandbox=""` iframe, which is the same treatment
`ui/static/tiling.js` already gives notebook output and the opposite of the
treatment Lab gets: Lab is a server we started behind a token we minted, and it
cannot function sandboxed. This can, so it does.

**Cached on the file's identity, not on a timer.** Rendering a large notebook is
a tenth of a second, the poll runs every two seconds, and the pane is redrawn
whenever anything about the notebook changes. Keying on mtime and size means a
render happens when the file has actually moved and never otherwise.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

from core import paths
from ui import tokens

log = logging.getLogger("grad.ui")

#: `(resolved path) -> (mtime, size, html)`. One entry per notebook; a workspace
#: with a hundred of them and every one opened is still a few megabytes, and the
#: alternative is re-rendering the same unchanged file on every redraw.
_CACHE: dict[str, tuple[float, int, str]] = {}


class NotAllowed(Exception):
    """The requested name is not a notebook in this workspace."""


def resolve(name: str) -> Path:
    """The notebook `name` refers to, or `NotAllowed`.

    The name arrives from an HTTP path on an unauthenticated local port, so it
    is untrusted input to a filesystem read. Two checks, because either alone
    leaves a hole: the name must have no directory part at all, and the resolved
    path must still be inside the notebooks directory once symlinks are
    followed. `..` is caught by the first, a symlink pointing out of the
    workspace only by the second.
    """
    if not name.endswith(".ipynb") or Path(name).name != name:
        raise NotAllowed(name)
    directory = paths.notebooks_dir().resolve()
    target = (directory / name).resolve()
    if directory not in target.parents or not target.is_file():
        raise NotAllowed(name)
    return target


def notebook_html(name: str) -> str:
    """A complete, script-free HTML document for one notebook."""
    path = resolve(name)
    stat = path.stat()
    cached = _CACHE.get(str(path))
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    body = _body(path)
    document = _document(name, body)
    _CACHE[str(path)] = (stat.st_mtime, stat.st_size, document)
    return document


def _body(path: Path) -> str:
    """The notebook as HTML, or a readable explanation of why not.

    A notebook that nbconvert refuses is usually one being written to as this
    reads it, which is an ordinary event when the agent is working -- so it
    produces a message in the pane and the next poll tries again, rather than an
    exception that takes the window down.
    """
    try:
        import nbformat  # noqa: PLC0415
        from nbconvert import HTMLExporter  # noqa: PLC0415
    except ImportError:
        return (
            "<p class='grad-render-note'>nbconvert is not installed. "
            "<code>pip install -e .[notebook]</code></p>"
        )
    try:
        notebook = nbformat.read(path, as_version=4)
        body, _ = HTMLExporter(template_name="basic").from_notebook_node(notebook)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.debug("could not render %s", path, exc_info=exc)
        return (
            "<p class='grad-render-note'>This notebook could not be rendered "
            f"({html.escape(type(exc).__name__)}). It may be mid-write; the next "
            "refresh will try again.</p>"
        )
    return body


def _document(name: str, body: str) -> str:
    """Wrap the fragment in the workspace's own typography.

    nbconvert's own stylesheet is Lab's, and dropping it into a pane that is
    otherwise this design would look like two applications in one window. The
    palette below is `ui/tokens.py`'s, read from it rather than copied, so the
    render cannot drift away from the rest of the app.
    """
    c = tokens.COLOUR
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: {c['paper']}; color: {c['ink']};
                font-family: {tokens.FONT_SANS}; font-size: 13px; }}
  body {{ padding: 14px 16px 40px; }}
  .cell {{ margin: 0 0 14px; }}
  .input_prompt, .output_prompt {{ display: none; }}
  pre, code, .highlight {{ font-family: {tokens.FONT_MONO}; font-size: 12px; }}
  .highlight, .input_area {{ background: {c['paper-raised']};
                             border: 2px solid {c['ink']}; }}
  .input_area pre, .highlight pre {{ margin: 0; padding: 9px 11px;
                                     overflow-x: auto; white-space: pre; }}
  .output_subarea {{ border-left: 2px solid {c['rule-mid']}; padding: 6px 0 6px 11px;
                     margin-top: 6px; }}
  .output_stderr pre {{ color: {c['broken-ink']}; background: {c['broken-tint']}; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  table td, table th {{ border: 1px solid {c['rule-soft']}; padding: 3px 7px; }}
  img, svg {{ max-width: 100%; height: auto; }}
  h1, h2, h3, h4 {{ font-family: {tokens.FONT_SANS}; margin: 18px 0 7px; }}
  a {{ color: {c['link']}; }}
  .grad-render-note {{ color: {c['muted']}; font-style: italic; }}
  .grad-render-head {{ position: sticky; top: 0; background: {c['ink']};
                       color: {c['paper']}; font-family: {tokens.FONT_MONO};
                       font-size: 11px; letter-spacing: .12em; padding: 5px 16px;
                       margin: -14px -16px 14px; }}
</style></head>
<body>
<div class="grad-render-head">READ ONLY · {html.escape(name)} · EDIT IN LAB</div>
{body}
</body></html>"""


def invalidate(name: str | None = None) -> None:
    """Drop a cached render. Called when a verify rewrites a notebook."""
    if name is None:
        _CACHE.clear()
        return
    try:
        _CACHE.pop(str(resolve(name)), None)
    except (NotAllowed, OSError):
        _CACHE.clear()
