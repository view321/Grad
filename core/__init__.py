"""Shared machinery behind the CLIs in `tools/`.

HANDOFF §7 requires exactly one write path to the ledger files and HANDOFF §8
requires every CLI to speak the same error envelope and exit codes. Both are
only true if there is one implementation, so it lives here rather than being
copied into each tool.
"""
