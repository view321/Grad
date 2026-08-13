"""KaTeX for the transcript (HANDOFF §10, "Known gaps").

`ui.markdown` has no KaTeX support, which is one of the two things a React app
would have given for free. The fix is small and lives here so it is written
once: load KaTeX plus its auto-render extension into the page head, then call
`renderMathInElement` on a container after each settled message.

Assets are served from `ui/assets/katex/` when present rather than from a CDN --
this is a desktop research tool and it should work offline. If the assets are
missing, `head_html()` falls back to the CDN and `assets_present()` reports
false so the app can say so out loud instead of silently rendering `$$` as text.
"""

from __future__ import annotations

from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets" / "katex"
CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

_DELIMITERS = """
    {left: '$$', right: '$$', display: true},
    {left: '\\\\[', right: '\\\\]', display: true},
    {left: '$', right: '$', display: false},
    {left: '\\\\(', right: '\\\\)', display: false}
"""


def assets_present() -> bool:
    return (ASSET_DIR / "katex.min.js").exists() and (ASSET_DIR / "katex.min.css").exists()


def _base() -> str:
    return "/katex" if assets_present() else CDN


def head_html() -> str:
    base = _base()
    return f"""
<link rel="stylesheet" href="{base}/katex.min.css">
<script defer src="{base}/katex.min.js"></script>
<script defer src="{base}/contrib/auto-render.min.js"></script>
<script>
window.gradRenderMath = function (selector) {{
    const el = document.querySelector(selector);
    if (!el || typeof renderMathInElement !== 'function') return;
    renderMathInElement(el, {{
        delimiters: [{_DELIMITERS}],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    }});
}};
</script>
"""


def install(app) -> None:
    """Register the head HTML and, when the assets are vendored, serve them."""
    from nicegui import ui

    ui.add_head_html(head_html())
    if assets_present():
        app.add_static_files("/katex", str(ASSET_DIR))


async def render(selector: str) -> None:
    """Run KaTeX over a container. Call once per settled message, not per token."""
    from nicegui import ui

    await ui.run_javascript(f"window.gradRenderMath && window.gradRenderMath({selector!r})", timeout=5.0)
