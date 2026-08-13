"""The four widgets that earn their keep (HANDOFF §10).

    "Generic chat UI is a solved, boring problem; the value is in surfacing this
     system's own state, which is otherwise invisible."

1. preflight panel      -- the checks as a live checklist, with the fix command
2. expectation vs outcome -- predicted band, actual marker, basis and comparability
3. quota and spend meter  -- measured tokens by stage, credits, rolling GPU spend
4. funnel view            -- 400 -> 50 -> 15 with stage-3's reason per survivor
"""

from ui.widgets.preflight_panel import preflight_panel
from ui.widgets.expectation_plot import expectation_panel
from ui.widgets.quota_meter import quota_meter, quota_panel
from ui.widgets.funnel_view import funnel_view

__all__ = ["preflight_panel", "expectation_panel", "quota_meter", "quota_panel", "funnel_view"]
