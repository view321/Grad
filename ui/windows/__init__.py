"""The eleven windows.

Each module here exposes `render(workspace)` and, optionally, `subtitle()` and
`chips()` for its title bar. None of them import each other, and none of them
read a ledger directly -- the data comes from `ui/models.py`, already shaped,
already caught. A window module is allowed to know about layout and pixels and
nothing else.

`ui/registry.py` is the list; adding a window means adding a module here and a
`WindowSpec` there.
"""

from __future__ import annotations

__all__ = [
    "chat",
    "editor",
    "evolve",
    "funnel",
    "ledger",
    "notebook",
    "papers",
    "preflight",
    "queue",
    "wiki",
]
