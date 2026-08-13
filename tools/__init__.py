"""The agent's custom capability, exposed as CLIs invoked over Bash (HANDOFF §8).

Each module here is a standalone CLI with `--json` on every subcommand, distinct
exit codes, and errors that state the fix. They are deliberately usable by a
human at a terminal, by `claude -p`, or by the NiceGUI app in `ui/` -- the UI
calls these rather than reimplementing their logic.
"""
