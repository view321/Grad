# Handoff: Grad — window system UI

## Overview

Grad is a personal autonomous ML-research agent (Python, Claude Agent SDK, NiceGUI desktop
app, embedded JupyterLab). This handoff covers the **visual system and the window layer**:
a tiling workspace shell plus eleven windows — notebook, agent chat, wiki, cited papers,
Evolve (ShinkaEvolve), LaTeX paper editor, ledger/expectations, preflight gates, quota &
budget, retrieval funnel, and run queue.

The design goal was Claude-app calm carrying a neo-brutalist structure with a light retro
note: cream paper ground, 2px ink rules, hard offset shadows on floating shells, monospace
for everything the machine produced, and exactly one accent colour per state.

## About the design files

The three `.dc.html` files in `reference/` are **design references created in HTML** —
prototypes of the intended look, not production code to copy. They open directly in a
browser (each is a self-contained page; fonts load from Google Fonts).

The implementation task is to **recreate these designs inside Grad's existing environment**:

- App chrome, panes, chat, ledger, quota, evolve, queue, funnel, preflight, papers, wiki →
  **NiceGUI** (Quasar under the hood) with a project stylesheet, using Grad's existing
  `ui/app.py` structure and `ui/widgets/*`.
- Notebook cell rendering → **a pinned JupyterLab theme extension** (see "JupyterLab" below).
  Do not restyle the iframe from the host page — cross-origin CSS will not apply.

Do not lift the HTML verbatim. Take the tokens, the measurements and the component anatomy.

## Fidelity

**High fidelity.** Colours, type, spacing, borders and states are final. Recreate
pixel-accurately. Content strings in the mock are placeholder research data — the real
values come from Grad's ledger, budget tracker and Jupyter kernel.

## Design tokens

### Colour

| Token | Hex | Role |
| --- | --- | --- |
| `ink` | `#14100C` | Text, all structural rules, dark bars, primary button fill |
| `paper` | `#F7F3E8` | Window surface |
| `paper-raised` | `#FFFDF8` | Cell bodies, input fields, message bubbles |
| `paper-sunk` | `#EFE8D8` | Title bars, gutters, side rails |
| `desk` | `#E8E2D4` | Page background behind window shells |
| `rule-soft` | `rgba(20,16,12,0.15)` | Hairline row/cell dividers |
| `rule-mid` | `rgba(20,16,12,0.2–0.3)` | Dashed secondary splits |
| `attention` | `#FFD400` | Needs the human: gates, unjudged, verify button, fix hints |
| `verified` | `#12A594` | Passing, in band, running-and-healthy |
| `verified-ink` | `#04302C` | Text on `verified` |
| `verified-tint` | `#DFF3EF` | Diff additions |
| `broken` | `#A3122F` | Errors, failed jobs, falsified expectations |
| `broken-tint` | `#FDEEF1` | Error output backgrounds, diff deletions |
| `broken-ink` | `#5A1020` / `#7A1024` | Text on `broken-tint` |
| `link` | `#B04A2C` | Links, function names in code |
| `muted` | `#8A8272` | Operators, secondary mono text |
| `muted-2` | `#9C9484` | Code comments |
| `literal` | `#0B7B6E` | Numbers and strings in code |
| `hatch-a` / `hatch-b` | `#F1EADA` / `#E7DFCC` | Figure placeholder hatching |

Rules: one accent per state, never two in the same element. Accents are **fills with ink
borders**, not gradients. Never tint a whole pane.

### Type

- `'Space Grotesk', Helvetica, sans-serif` — 500/700 — all UI text and prose.
- `'JetBrains Mono', monospace` — 400/700 — anything the machine produced or that must
  align: code, IDs, numbers, timings, labels, status chips, table bodies.
- `'Instrument Serif', Georgia, serif` — 400 + italic — markdown headings inside notebooks,
  paper titles, LaTeX preview, math variables.

Sizes in use: 9, 10, 11, 12, 12.5, 13, 13.5, 14, 14.5, 15, 19, 21, 28, 30, 34 px.
Uppercase mono labels carry `letter-spacing: 0.08–0.22em`.
Body line-height 1.55–1.62; code 1.65–1.75.

### Structure

- Structural border: `2px solid #14100C`. Hairline: `1px solid rgba(20,16,12,0.15)`.
  Secondary: `1px dashed rgba(20,16,12,0.3)`. Empty/pending: `2px dashed #14100C`.
- **Radius: 0 everywhere.**
- Shadow: `8px 8px 0 #14100C` (workspace shell), `6px 6px 0 #14100C` (standalone window
  cards). No blur shadows anywhere.
- Cell state stripe: `6px` solid left border — `#12A594` ok, `#A3122F` error, `#FFD400`
  awaiting, `#14100C` focused/empty.
- Pane split handle: `8px` wide, `#14100C`, three centred `2px` `#F7F3E8` dots,
  `cursor: col-resize`.
- Padding scale: 6/7 px (chips), 9–11 px (rows, buttons), 12–14 px (panels),
  16–22 px (cell bodies), 20 px (page gutter).
- Gutter column in notebooks: `72px` in panes, `76–84px` standalone.

### States (all components)

- Hover on ghost controls: background `#EFE8D8`.
- Active/selected tab or chip: background `#14100C`, text `#F7F3E8`.
- Primary action: `#FFD400` fill + 2px ink border.
- Destructive/blocked action: `#A3122F` fill, white text.
- Disabled: background `#EFE8D8`, `opacity: 0.5`, no pointer events.
- Focus-visible: `outline: 2px solid #14100C; outline-offset: 2px`.
- Blinking caret / live indicator: 7–8px ink or accent block,
  `@keyframes gradblink { 0%,49% {opacity:1} 50%,100% {opacity:0} }`, `1.1s steps(1) infinite`.

## Workspace shell

File: `reference/Grad Workspace.dc.html`. Design width 1720px content, 900px pane height.

1. **Title bar** (`#14100C`, `#F7F3E8` text, 2px bottom rule, ~42px):
   `∇` mark in a `#FFD400` 22px square with a 2px paper border, `GRAD` at 15px mono
   `letter-spacing: .22em`, then a `#F7F3E8` 2px divider; project selector; agent run state
   chip (`#12A594` fill, ink square dot, "AGENT RUNNING · step 14") plus `■ PAUSE`; spacer;
   session quota strip (150×12px, 1.5px paper border, segments `#FFD400` used-by-chat and
   `#12A594` used-by-tools) with `$4.12 / $8.00` and reset countdown; `⌘K` and `LAYOUTS ▾`.
2. **Window opener strip** (`#EFE8D8`, 2px bottom rule, ~30px): one 11px mono uppercase cell
   per window, separated by 1px hairlines. Open windows render inverted (ink fill, paper
   text). Right side shows layout shortcuts.
3. **Tiling area**: horizontal flex of panes separated by 8px drag handles. Panes are
   independently resizable; a pane may split vertically (the right pane stacks LEDGER over
   QUOTA, each keeping its own title bar).
4. **Status bar** (`#14100C`, 11px mono, ~30px): cwd, kernel, queue/gpu counts, spacer,
   `⌥drag to retile`, and an `#FFD400` chip with the open-window count.

**Pane title bar** (every window, ~30px, `#EFE8D8`, 2px bottom rule): 11px mono uppercase
name at `letter-spacing: .14em`; a 55%-opacity mono subtitle; optional state chips;
spacer; `⇱ ⇲ ✕` at 50% opacity.

## Windows

### 1. Notebook (JupyterLab iframe)

Standalone reference: `reference/Notebook Paper.dc.html`.

- **Toolbar** (2px bottom rule): a joined button group — `▶ RUN` (active: ink fill),
  `▶▶ ALL`, `■ STOP`, `↻ RESTART`, each 2px ink border with `border-left: 0` on the joins;
  then `✓ VERIFY — FRESH KERNEL` on `#FFD400`; spacer; `ruler 88` in a dashed box;
  `↗ OPEN IN LAB`.
- **Verify banner** (`#12A594`, `#04302C` text, 2px bottom rule): `NB VERIFY` label, the
  result sentence ("clean — 12 cells ran top to bottom on a fresh kernel · 41.8s · date"),
  and a `CITABLE` chip (`#04302C` fill, `#12A594` text). Failure variant: `#A3122F` fill,
  white text, chip reads `NOT CITABLE`. Stale variant: `#FFD400`, `RE-VERIFY`.
- **Cells**: CSS grid `72–84px 1fr`. Gutter is `#EFE8D8` with a 2px right rule and holds
  `In [n]:` (12px mono, right-aligned) and, under it, an execution-time chip
  (10px mono, ink fill, paper text). Body has the 6px state stripe.
  - Markdown cell: gutter reads `MD` at 50% opacity; body uses Instrument Serif 30px
    headings and 15px Space Grotesk prose at `max-width: 74ch`; display math sits in a
    2px-bordered `#FFFDF8` box.
  - Code cell: `pre`, 13.5px JetBrains Mono, line-height 1.7, token colours above.
  - Output: a `56px` mono label column (`STDOUT`, `OUT[n]`, `ERR`) separated by a 1px
    dashed rule.
  - Table output: ink `thead` with paper text; 1px hairline rows; every other row
    `#FDFAF2`; verdict chips in the last column (`IN BAND` teal, `HIGH · UNJUDGED` yellow,
    `OUT OF BAND` outlined).
  - Figure output: 2px ink frame over 45° hatching, a centred caption chip, and a
    `640×400 · PNG` tag pinned bottom-left in ink.
  - Error output: `#FDEEF1` background, `#A3122F` stripe and label, an
    `ErrorType — cell n of m` chip, the traceback in `#5A1020`, and a `FIX` box —
    2px ink border on `#FFD400` — containing the exact shell command that repairs it.
  - Empty trailing cell: dashed-ink stripe and a blinking caret.
- **Add-cell row**: `+ CODE` / `+ MARKDOWN` as 2px dashed ink buttons at 75% opacity.
- **Footer** (ink): `CMD`, cursor position, cell/output counts, spacer, `kernel owner: lab`,
  and a `#12A594` project chip.

### 2. Agent chat

- User message: right-aligned, `max-width: 88%`, 2px ink border on `#FFFDF8`, 11px padding,
  14px text. Role line above in 10px mono at 50% opacity.
- Grad message: left, avatar = 16px `#FFD400` square with 1.5px ink border and `∇`; body
  indented `23px`; inline code gets an `#EFE8D8` background.
- **Expectation card**: 2px ink border; `#FFD400` header bar (`EXPECTATION REGISTERED` +
  id); body is a 12px mono key/value list — claim, band, falsifier, source.
- **Tool call**: ink header bar (`TOOL`, tool name, args, right-aligned result chip
  `OK 8.4s` in teal); body `#FFFDF8` with output and a `▸ n more output lines` disclosure.
- **Streaming row**: 2px dashed ink box, blinking square, "running cell 4 of 12 …",
  `esc to interrupt` at right.
- **Gate card**: 2px `#A3122F` border, solid `#A3122F` header (`GATE — YOUR CALL`), a
  sentence naming the exact cost and resource, then `✓ APPROVE` (teal), `✎ EDIT PLAN`,
  `✕ DENY` — all 2px ink borders.
- **Composer**: mode chips (`ASK` active ink / `PLAN` / `RUN`), `@notebook @paper @wiki`
  mention hint, a 2px-bordered field on `#FFFDF8` with a blinking caret, and a `SEND ⏎`
  button on `#FFD400`.

### 3. Wiki + references

Two panes split by an 8px handle: chat on the left (same message anatomy as above; answers
carry superscript reference markers in `#B04A2C`), references rail on the right (`#EFE8D8`,
440px). Each reference: a numbered ink chip, file path, line range, and the snippet in a
1.5px-bordered `#FFFDF8` `pre` — bordered `#A3122F` on `#FDEEF1` when the snippet is the
faulty one. Answer actions: `→ OPEN IN EDITOR` (yellow), `→ ASK GRAD TO PATCH`.

### 4. Cited papers

List + reader. Filter chips in the title bar (`CITED IN PAPER` active, `READ`, `QUEUED`).
Each row: 70×92px cover placeholder (1.5px border, horizontal stripe fill; dashed border
when unread), Instrument Serif 21px title, 11px mono authors/arXiv line, then status chips
— `3 CLAIMS DEPEND ON THIS` (yellow), figure/table references (outlined),
`CONTRADICTS exp-…` (teal), `QUEUED BY GRAD · not read` (dashed). Selected row gets a 6px
ink stripe and `#FFFDF8` fill. Reader rail (520px, `#EFE8D8`): page frame plus a `#FFD400`
`PULLED INTO exp-…` card holding the sentence that became an expectation.

### 5. Evolve (ShinkaEvolve)

Three panes: population stats (300px, `#EFE8D8`, mono key/value rows: islands, migrations,
novelty, spend; a dashed box restating the objective); lineage chart (bars per generation,
1.5px ink borders — `#EFE8D8` ordinary, `#FFD400` new best, `#12A594` current champion,
axis captions in 10px mono); champion diff (420px, `#FFFDF8`): a header with the fitness
delta chip (teal), a unified diff with `#FDEEF1`/`#7A1024` deletions and
`#DFF3EF`/`#04302C` additions, then `✓ ADOPT INTO MAIN` (yellow) and `→ SEND TO NOTEBOOK`.
Title bar carries an `EVOLVING` chip and `■ HALT`.

### 6. Paper editor (LaTeX)

Three panes: outline (190px, `#EFE8D8`; active section inverted ink; a `#A3122F`-bordered
`#FDEEF1` warning box — "2 claims uncited. Grad blocks the build until each is bound to a
run or a paper"); source (`pre`, 12.5px mono, `\gradcite{run-…}` / `\gradexp{exp-…}` macros
highlighted, uncited sentences flagged `#FDEEF1`); preview (`#FFFDF8`, Instrument Serif
headings, justified 14.5px prose, the uncited sentence underlined 2px `#A3122F`, matted
figure with caption). Title bar: `⌘S SAVE`, `BUILD PDF` (yellow).

### 7. Ledger / expectations

Filter chips (`OPEN n` active, `MET n`, `BROKEN n`). Each entry: 6px left stripe by state
(`#FFD400` open, `#12A594` met, `#A3122F` broken), id + state chip + timestamp, the claim
in 13.5px text, and — for open entries — a **band strip**: 30px tall, 1.5px ink border on
`#FFFDF8`, the predicted band as a 35%-opacity `#12A594` block, the observed value as a 2px
ink tick with its number above, falsifier bounds as `#A3122F` ticks, and min/band/max
labels in 10px mono underneath.

### 8. Quota & budget

Window-level 5-hour meter: 22px bar, 2px ink border, `#FFD400` chat segment (labelled
inside), `#12A594` tool segment, `#FFFDF8` remainder; legend in 10px mono.
Spend today: per-model horizontal bars (ink = sonnet, `#B04A2C` = opus, `#12A594` = gpu)
with dollar values. Always ends with the **honesty note** in a 2px dashed box: token counts
are Grad's own tally, not the provider's — an estimate within ±5%.

### 9. Preflight + gates

Checklist rows, 1px hairline separated. Each: an 18px status square with 2px ink border —
`#12A594` ✓, `#FFD400` ! (row background `#FFFBE8`), `#A3122F` ✕ with white glyph — then
the check sentence and a right-aligned detail or `FIX` button. Footer: `▶ PROCEED`
(disabled while anything blocks), the one-click remedy on `#FFD400`, and a blocking count.

### 10. Funnel (retrieval)

Stacked stage bars, each 34px with a 2px ink border, progressively indented and narrowed:
`CORPUS · n chunks` (`#EFE8D8`) → `BM25 + EMBED → n` → `RERANK → n` (`#FFD400`) →
`IN CONTEXT n` (`#12A594`). Below a dashed rule, the surviving chunks in rank order with
scores; dropped chunks at 45% opacity with the reason.

### 11. Run queue / GPU jobs

Full-width table: ink `thead`; columns job, what, device, progress, eta, cost, state.
Progress is a 12px bar with a 1.5px border — teal fill running, ink fill done, `#A3122F`
border+fill failed, dashed empty when queued. State chips: `RUNNING` teal,
`WAITING GATE` yellow, `DONE` outlined, `FAILED · KeyError` crimson. Running row is
`#FFFDF8`.

## Interactions & behaviour

- **Tiling**: panes resize by dragging the 8px handles; `⌥`+drag a title bar to retile;
  `⌥1/⌥2/⌥3` switch tile/stack/full. Persist layout per project. Minimum pane width 320px.
- **Window opener**: clicking a name opens it into the focused pane (or splits if the pane
  already holds one); clicking an open name closes it.
- **Notebook**: Run/Run-all/Stop/Restart map to Jupyter kernel commands. `VERIFY` runs the
  notebook top-to-bottom on a fresh kernel and rewrites the verify banner; the banner is
  the sole source of the citable/not-citable state, and it goes stale (yellow) on any edit.
- **Gates**: approving a gate resumes the agent loop; denying returns control to chat with
  the reason attached. A gate always states the exact spend and resource before the ask.
- **Preflight**: `PROCEED` is disabled while any ✕ row exists; the yellow remedy button
  performs the fix and re-runs the checklist in place.
- **Ledger**: a new run redraws the band strip's tick; crossing a falsifier bound flips the
  entry to `BROKEN` (crimson) and posts a message into chat.
- **Evolve**: lineage bars append per generation; adopting a champion opens the diff as a
  patch against main.
- **Motion**: only two — the 1.1s step blink for carets/live indicators, and instant state
  swaps. No easing curves, no fades, no skeleton shimmer. Progress bars update in place.

## State (host side)

Per window: `open`, `pane_id`, `size_fraction`, `focused`. Global: `project`,
`agent_state` (idle | running | awaiting_gate | paused), `session_spend`, `quota_window`,
`queue`, `kernel_state`, `verify_state` per notebook, `ledger_entries`, `evolve_run`.
Layout persists to disk per project; everything else is live from the agent loop.

## JupyterLab

The notebook window is a real Lab iframe, so its interior cannot be styled from NiceGUI.
Implement it as a **pinned JupyterLab theme extension** applying the tokens above to Lab's
own CSS variables and cell classes — `--jp-layout-color0/1/2`, `--jp-border-color*`,
`--jp-cell-editor-background`, `--jp-code-font-family`, `--jp-content-font-family`,
`.jp-InputArea-prompt`, `.jp-OutputArea-output`, `.jp-RenderedText`. Set
`"theme": "Grad Paper"` in `config/jupyter/overrides.json` in place of JupyterLab Dark, and
keep `notebook-extension` defaults (ruler at 88) as they are. Chrome that Grad owns — the
toolbar, verify banner, and footer — stays in NiceGUI **above** the iframe, styled
identically, so the seam is invisible.

## Assets

None. Every mark is CSS: the `∇` wordmark glyph, `▶ ▶▶ ■ ↻ ✓ ✕ ! ⇱ ⇲ ↗ ⏎ ▾ ▸` from the
system font, hatched figure placeholders from `repeating-linear-gradient(135deg, …)`, and
paper covers from a horizontal stripe gradient. Fonts: Space Grotesk, JetBrains Mono,
Instrument Serif (Google Fonts — vendor them locally for an offline desktop app).

## Files

- `reference/Grad Workspace.dc.html` — the tiling shell with chat, notebook, ledger and
  quota tiled live.
- `reference/Grad Windows.dc.html` — wiki, papers, evolve, editor, preflight, funnel, queue.
- `reference/Notebook Paper.dc.html` — the notebook window at full size; the reference for
  the JupyterLab theme extension.
