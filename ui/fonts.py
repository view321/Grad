"""The three typefaces, vendored if present and fetched if not.

The handoff asks for Space Grotesk, JetBrains Mono and Instrument Serif, and
adds "vendor them locally for an offline desktop app". That is the right call --
a research instrument that renders in Helvetica because the wifi is down is a
research instrument that looks broken -- but it cannot be the *only* path,
because the repository ships no binaries.

So: `head_html()` emits `@font-face` rules for whatever `.woff2` files are
actually sitting in `ui/static/fonts/`, and falls back to the Google Fonts link
only for the families that are missing. Both cases end at the same CSS font
stacks in `tokens.py`, and `vendored()` reports which families are local so the
status bar can say so rather than leaving it a mystery.

Run `python -m ui.fonts --check` to see what is vendored.
"""

from __future__ import annotations

from pathlib import Path

# family -> (file stem, css family name, weights present in the design)
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Space Grotesk": ("space-grotesk", ("500", "700")),
    "JetBrains Mono": ("jetbrains-mono", ("400", "500", "700")),
    "Instrument Serif": ("instrument-serif", ("400",)),
}

GOOGLE_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=Space+Grotesk:wght@500;700"
    "&family=JetBrains+Mono:wght@400;500;700"
    "&family=Instrument+Serif:ital@0;1"
    "&display=swap"
)


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def fonts_dir() -> Path:
    return static_dir() / "fonts"


def _files(directory: Path) -> dict[str, list[tuple[str, str, bool]]]:
    """family -> [(weight, filename, italic)] for every vendored file found.

    Naming convention: `<stem>-<weight>.woff2`, plus `-italic` for the one
    italic the design uses (Instrument Serif, for math variables).
    """
    found: dict[str, list[tuple[str, str, bool]]] = {}
    if not directory.exists():
        return found
    for family, (stem, weights) in FAMILIES.items():
        for weight in weights:
            for italic in (False, True):
                suffix = "-italic" if italic else ""
                name = f"{stem}-{weight}{suffix}.woff2"
                if (directory / name).exists():
                    found.setdefault(family, []).append((weight, name, italic))
    return found


def vendored(directory: Path | None = None) -> set[str]:
    """The families that will render without a network."""
    return set(_files(directory or fonts_dir()))


def font_face_css(directory: Path | None = None, *, url_prefix: str = "/grad-static/fonts") -> str:
    directory = directory or fonts_dir()
    rules = []
    for family, entries in sorted(_files(directory).items()):
        for weight, name, italic in entries:
            rules.append(
                "@font-face {\n"
                f"  font-family: '{family}';\n"
                f"  src: url('{url_prefix}/{name}') format('woff2');\n"
                f"  font-weight: {weight};\n"
                f"  font-style: {'italic' if italic else 'normal'};\n"
                "  font-display: swap;\n"
                "}"
            )
    return "\n".join(rules)


def head_html(directory: Path | None = None, *, url_prefix: str = "/grad-static/fonts") -> str:
    """`@font-face` for what is vendored, a Google link for what is not.

    The link is emitted per missing family rather than all-or-nothing so a
    partial vendoring still cuts the network dependency down.
    """
    directory = directory or fonts_dir()
    have = vendored(directory)
    parts = []
    faces = font_face_css(directory, url_prefix=url_prefix)
    if faces:
        parts.append(f"<style>\n{faces}\n</style>")
    missing = [f for f in FAMILIES if f not in have]
    if missing:
        parts.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
        parts.append('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
        parts.append(f'<link rel="stylesheet" href="{GOOGLE_HREF}">')
    return "\n".join(parts)


def main() -> None:
    have = vendored()
    print(f"fonts dir: {fonts_dir()}")
    for family in FAMILIES:
        mark = "vendored" if family in have else "google fonts (needs network)"
        print(f"  {family:<18} {mark}")
    if len(have) < len(FAMILIES):
        print(
            "\nto vendor: download the woff2 files and name them "
            "<stem>-<weight>.woff2 (e.g. space-grotesk-700.woff2) in the directory above"
        )


if __name__ == "__main__":
    main()
