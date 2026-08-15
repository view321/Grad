"""JupyterLab server configuration for the embedded Lab tab (HANDOFF-2 §19).

Two settings here are the difference between "the tab works" and "an afternoon
of wondering why the iframe is blank".

**Framing.** JupyterLab ships `X-Frame-Options: SAMEORIGIN` and a
`frame-ancestors 'self'` CSP, both of which block embedding from another origin.
Lab runs on its own port, so the Grad app *is* another origin. The headers below
override both for the app's origin only -- not for `*`, which would let any page
frame a server that can execute code as this user.

`X-Frame-Options` has no origin list (it is SAMEORIGIN or DENY), so it is
cleared explicitly in the headers dict and the CSP does the scoping. It is
cleared *here* rather than via `xheaders`: `xheaders` controls whether Tornado
trusts `X-Forwarded-*` / `X-Real-Ip` from a proxy and has nothing to do with
framing. Setting `xheaders = False` and expecting the frame header to disappear
leaves Jupyter emitting `SAMEORIGIN`, and the iframe stays blank in every
browser that honours it -- the exact failure this file exists to prevent.

**Origin checks.** A cross-origin websocket from the app's port is rejected by
default, which breaks the kernel connection while the page still renders. That
is the confusing half of the failure, so it is set alongside the headers rather
than discovered later.

Nothing here weakens the token: Lab is still started with one, on 127.0.0.1
only, by `tools/lab.py`.
"""

import os
import re

# The Grad UI's origin. `tools/lab.py` sets this when it launches the server, so
# the two cannot drift apart; the default matches ui/app.py's default port.
_APP_ORIGIN = os.environ.get("GRAD_UI_ORIGIN", "http://127.0.0.1:8080")
_LAB_PORT = os.environ.get("GRAD_LAB_PORT", "8889")

c = get_config()  # noqa: F821 - injected by the Jupyter config loader

c.ServerApp.ip = "127.0.0.1"
c.ServerApp.open_browser = False
c.ServerApp.port = int(_LAB_PORT)

# `127.0.0.1:8080` and `localhost:8080` are different origins to a browser, and
# which one the app is opened on is not something this file gets to decide. The
# alias is only added when the configured origin actually carries a numeric
# port: `rsplit(':', 1)[-1]` on a portless origin like `http://example.com`
# yields `example.com`, and `http://localhost:example.com` is an invalid source
# that browsers drop silently -- narrowing the allowed ancestors instead of
# widening them.
_PORT = _APP_ORIGIN.rsplit(":", 1)[-1]
_LOCALHOST_ALIAS = f" http://localhost:{_PORT}" if _PORT.isdigit() else ""

# Framing, scoped to the app's origin rather than to `*`.
c.ServerApp.tornado_settings = {
    "headers": {
        "Content-Security-Policy": f"frame-ancestors 'self' {_APP_ORIGIN}{_LOCALHOST_ALIAS}",
        # Cleared deliberately; the CSP above does the scoping. See the module
        # docstring for why this is not `xheaders`.
        "X-Frame-Options": "",
    },
}
# The same pair of hosts as the CSP above, and for the same reason. `allow_origin`
# takes exactly one origin, so setting it to `127.0.0.1:<port>` rejects the
# websocket when the app is opened on `localhost:<port>` -- and *that* failure is
# the confusing half: the page renders, the frame loads, and only the kernel
# connection dies. `allow_origin_pat` is the regex form, which is how both can be
# named without widening this to `*`.
_ORIGINS = [_APP_ORIGIN] + ([_LOCALHOST_ALIAS.strip()] if _LOCALHOST_ALIAS else [])
c.ServerApp.allow_origin_pat = "|".join(re.escape(origin) for origin in _ORIGINS)
c.ServerApp.allow_credentials = True

# The websocket the kernel connection rides on.
c.ServerApp.allow_remote_access = False
c.ServerApp.disable_check_xsrf = False

# Kernel ownership discipline (§19): Lab has its own kernel manager and
# `tools/nb.py` spawns its own detached kernels. Two owners over one notebook
# reproduces the "works in the kernel that grew it" failure `nb verify` exists
# to catch, so the rule is unchanged -- anything edited here passes
# `python -m tools.nb verify <path> --json` before it is cited in notes/ or
# referenced from a ledger entry. The Verify button in the Notebooks tab is
# the enforcement surface for that.
