"""The NiceGUI desktop interface (HANDOFF §10).

The UI stays thin on purpose: it transports events, renders state, and calls the
CLIs from §8. It holds no logic of its own. Anything the UI can do, the CLIs can
already do, which keeps the terminal path alive and keeps the portability claim
honest.
"""
