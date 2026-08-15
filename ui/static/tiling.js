/* The parts of the window system that cannot go through the websocket.
 *
 * NiceGUI's model is a server-side element tree where every mutation is a round
 * trip. That is the right trade for almost all of this app and the wrong one for
 * exactly three things, which is what lives here:
 *
 *   1. Pane resizing. A `mousemove` handler that asked Python for a new width
 *      would run at the speed of the socket. Instead the drag writes
 *      `--grad-fraction` straight onto the DOM and only the *settled* fractions
 *      are emitted back, once, on mouseup.
 *   2. The Lab iframe. Browsers destroy and recreate an <iframe> when it is
 *      reparented, so moving the notebook window between panes would reload
 *      JupyterLab -- kernel, scroll position, unsaved cells and all. The iframe
 *      therefore lives in a fixed overlay outside the pane tree and is flown to
 *      wherever its anchor currently is.
 *   3. Dragging a title bar to retile. The drop indicator has to track the
 *      pointer, and the hit test needs the live geometry of every pane -- both
 *      at frame rate. Only the settled drop is sent back, as one event.
 *   4. Modifier chords. Alt+1/2/3 is keyboard state the server never sees.
 *
 * Everything else -- what is open, what is focused, what persists -- stays in
 * Python, where it can be tested.
 */
(() => {
  'use strict';
  if (window.__gradTilingLoaded) return;
  window.__gradTilingLoaded = true;

  const MIN_PANE_PX = 320;

  const emit = (name, data) => {
    if (typeof window.emitEvent === 'function') window.emitEvent(name, data);
  };

  const fractionsOf = (items, sizes, totalPx) => {
    // Clamp to the same floor Python enforces, so a drag cannot produce a
    // layout the server would immediately rewrite underneath the cursor.
    const floor = totalPx > 0 ? Math.min(0.5, MIN_PANE_PX / totalPx) : 0.06;
    const clamped = sizes.map((s) => Math.max(floor, s));
    const total = clamped.reduce((a, b) => a + b, 0) || 1;
    return clamped.map((s) => s / total);
  };

  /* ---------------------------------------------------------------- drag */
  const startDrag = (handle, event) => {
    const vertical = handle.classList.contains('row');
    const parent = handle.parentElement;
    if (!parent) return;
    const panes = Array.from(parent.children).filter((c) => !c.classList.contains('grad-handle'));
    const index = Array.from(parent.children).indexOf(handle);
    // The two panes this handle sits between.
    const before = Array.from(parent.children).slice(0, index).filter((c) => !c.classList.contains('grad-handle')).pop();
    const after = Array.from(parent.children).slice(index + 1).find((c) => !c.classList.contains('grad-handle'));
    if (!before || !after) return;

    const size = (el) => (vertical ? el.getBoundingClientRect().height : el.getBoundingClientRect().width);
    const startPos = vertical ? event.clientY : event.clientX;
    const startBefore = size(before);
    const startAfter = size(after);
    const totalPx = panes.reduce((sum, p) => sum + size(p), 0);

    handle.classList.add('dragging');
    // The same class the *retile* drag sets, and for the reason recorded beside
    // its rule in `ui/tokens.py`: a cross-origin iframe hit-tests the pointer
    // before this document does. Only the title-bar drag was setting it, and
    // this is the drag that starts *adjacent* to a pane.
    //
    // The inline cursor below outranks the class's `grabbing`, so the resize
    // cursor is still the one shown.
    document.body.classList.add('grad-dragging');
    document.body.style.cursor = vertical ? 'row-resize' : 'col-resize';

    /* Pointer capture, and listeners on the *handle* rather than the document.
     *
     * A drag has exactly one way to end correctly and several ways to end
     * badly, and the bad ones all used to leave the same wreckage: the move
     * listener still attached, `grad-dragging` still on the body, the resize
     * cursor still showing, and the next stray movement going on resizing with
     * no button held. Listening on the document catches none of them, because
     * the events stop arriving at the document:
     *
     *   - the pointer crosses into the embedded Lab, which hit-tests first;
     *   - it leaves the window entirely and the button comes up outside;
     *   - the window loses focus mid-drag, or the OS cancels the gesture.
     *
     * `setPointerCapture` retargets every subsequent pointer event for this
     * pointer to the handle until it is released, which covers the first two by
     * construction -- capture outranks hit-testing, and events keep arriving
     * past the window edge. `pointercancel` and `lostpointercapture` cover the
     * third: whatever takes the gesture away, one of them fires, and both run
     * the same teardown. `finish` is idempotent because releasing capture
     * inside it raises `lostpointercapture` re-entrantly.
     */
    let done = false;
    try {
      handle.setPointerCapture(event.pointerId);
    } catch (err) {
      // No capture available (a synthetic event, an ancient engine). The
      // listeners below still work; only the iframe and out-of-window cases
      // degrade to the old behaviour, which `finish` then cleans up on blur.
    }

    const onMove = (moveEvent) => {
      const delta = (vertical ? moveEvent.clientY : moveEvent.clientX) - startPos;
      const nextBefore = startBefore + delta;
      const nextAfter = startAfter - delta;
      if (nextBefore < MIN_PANE_PX || nextAfter < MIN_PANE_PX) return;
      before.style.setProperty('--grad-fraction', (nextBefore / totalPx).toFixed(6));
      after.style.setProperty('--grad-fraction', (nextAfter / totalPx).toFixed(6));
      reflowFrames();
    };

    const finish = (endEvent) => {
      if (done) return;
      done = true;
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', finish);
      handle.removeEventListener('pointercancel', finish);
      handle.removeEventListener('lostpointercapture', finish);
      window.removeEventListener('blur', finish);
      try {
        if (handle.hasPointerCapture?.(event.pointerId)) {
          handle.releasePointerCapture(event.pointerId);
        }
      } catch (err) {
        /* already released with the element, or never captured */
      }
      handle.classList.remove('dragging');
      document.body.classList.remove('grad-dragging');
      document.body.style.cursor = '';
      emit('grad_resize', {
        axis: vertical ? 'slots' : 'columns',
        column: vertical ? Number(parent.dataset.columnIndex || 0) : null,
        fractions: fractionsOf(panes, panes.map((p) => size(p)), totalPx),
        total_px: Math.round(totalPx),
      });
    };

    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', finish);
    handle.addEventListener('pointercancel', finish);
    handle.addEventListener('lostpointercapture', finish);
    window.addEventListener('blur', finish);
    event.preventDefault();
  };

  document.addEventListener('pointerdown', (event) => {
    // Primary button only: a right-click on a divider is a context menu, not a
    // resize that never ends because no `pointerup` for button 2 is coming.
    if (event.button !== 0) return;
    const handle = event.target.closest?.('.grad-handle');
    if (handle) startDrag(handle, event);
  });

  /* ------------------------------------------ drag a title bar to retile */
  /* Visual Studio's gesture, under this app's constraint: every pane stays
   * visible, so a drop has to name a *position* -- which column, and where in it
   * -- not just a container. Three outcomes, chosen by where the pointer is:
   *
   *   over another window's title bar -> swap the two panes
   *   near a column's left/right edge -> a new column at that edge
   *   anywhere else over a column     -> insert at the nearest slot boundary
   *
   * The indicator is painted from the same hit test that produces the emitted
   * event, so the line drawn is exactly what the server is asked for. Anything
   * else and the drop lands somewhere the user was not shown.
   */
  const DRAG_THRESHOLD_PX = 5;  // under this a press is a click, not a drag
  const EDGE_RATIO = 0.22;      // of a column's width, at each side
  const EDGE_MAX_PX = 90;
  const MAX_COLUMNS = 3;        // mirrors ui/layout.py

  let drag = null;
  let indicator = null;

  const columnsOf = () => {
    const tiles = document.querySelector('.grad-tiles');
    return tiles ? Array.from(tiles.querySelectorAll(':scope > .grad-column')) : [];
  };
  const slotsOf = (column) => Array.from(column.querySelectorAll(':scope > .grad-slot'));
  const windowOf = (slot) => slot.querySelector('.grad-titlebar[data-window]')?.dataset.window;

  /** Where a window sits right now, read from the DOM rather than remembered:
   *  the pane tree is rebuilt by the server on every retile. */
  const positionOf = (id) => {
    const columns = columnsOf();
    for (let c = 0; c < columns.length; c += 1) {
      const slots = slotsOf(columns[c]);
      for (let s = 0; s < slots.length; s += 1) {
        if (windowOf(slots[s]) === id) return { column: c, slot: s, alone: slots.length === 1 };
      }
    }
    return null;
  };

  const hitTest = (x, y, self) => {
    const columns = columnsOf();
    if (!columns.length) return null;

    // 1. another window's title bar -> swap.
    const under = document.elementFromPoint(x, y);
    const bar = under?.closest?.('.grad-titlebar[data-window]');
    if (bar && bar.dataset.window !== self) return { kind: 'swap', other: bar.dataset.window };

    // 2. which column? Outside them all, the nearest one, remembering which side
    //    -- a drop past the right edge is how you make a column, so it must not
    //    be clamped into meaning "drop inside the last one".
    let index = columns.findIndex((c) => {
      const b = c.getBoundingClientRect();
      return x >= b.left && x < b.right;
    });
    let outside = 0;
    if (index < 0) {
      outside = x < columns[0].getBoundingClientRect().left ? -1 : 1;
      index = outside < 0 ? 0 : columns.length - 1;
    }
    const box = columns[index].getBoundingClientRect();

    // 3. the edge bands -> a new column beside this one. The cap is counted the
    //    way layout.py counts it: a window dragged out of a column it holds
    //    alone takes that column with it, so the move can create one without
    //    ever exceeding MAX_COLUMNS.
    const here = positionOf(self);
    const effective = columns.length - (here && here.alone ? 1 : 0);
    const band = Math.min(EDGE_MAX_PX, box.width * EDGE_RATIO);
    const left = outside < 0 || (outside === 0 && x < box.left + band);
    const right = outside > 0 || (outside === 0 && x > box.right - band);
    if ((left || right) && effective < MAX_COLUMNS) {
      return {
        kind: 'column',
        column: left ? index : index + 1,
        rect: { left: left ? box.left : box.right, top: box.top, height: box.height },
      };
    }

    // 4. otherwise the slot boundary nearest the pointer. `slot` is a boundary
    //    index into the column as it looks *now*, dragged window included;
    //    layout.py corrects for the pull when the move stays in one column.
    const slots = slotsOf(columns[index]);
    let slot = slots.length;
    for (let i = 0; i < slots.length; i += 1) {
      const b = slots[i].getBoundingClientRect();
      if (y < b.top + b.height / 2) { slot = i; break; }
    }
    const edge = slot < slots.length
      ? slots[slot].getBoundingClientRect().top
      : (slots.length ? slots[slots.length - 1].getBoundingClientRect().bottom : box.top);
    return { kind: 'slot', column: index, slot, rect: { left: box.left, top: edge, width: box.width } };
  };

  const clearPaint = () => {
    document.querySelectorAll('.grad-swap-target').forEach((el) => el.classList.remove('grad-swap-target'));
    if (indicator) indicator.style.display = 'none';
  };

  const paint = (hit) => {
    clearPaint();
    if (!hit) return;
    if (hit.kind === 'swap') {
      const bar = document.querySelector(`.grad-titlebar[data-window="${CSS.escape(hit.other)}"]`);
      if (bar) bar.classList.add('grad-swap-target');
      return;
    }
    if (!indicator) {
      indicator = document.createElement('div');
      document.body.appendChild(indicator);
    }
    const vertical = hit.kind === 'column';
    indicator.className = `grad-drop-indicator${vertical ? ' vertical' : ''}`;
    indicator.style.display = 'block';
    indicator.style.left = `${Math.round(hit.rect.left) - (vertical ? 2 : 0)}px`;
    indicator.style.top = `${Math.round(hit.rect.top) - (vertical ? 0 : 2)}px`;
    indicator.style.width = vertical ? '4px' : `${Math.round(hit.rect.width)}px`;
    indicator.style.height = vertical ? `${Math.round(hit.rect.height)}px` : '4px';
  };

  const endDrag = (commit) => {
    if (!drag) return;
    const { active, hit, id, ghost } = drag;
    drag.bar?.classList.remove('grad-drag-source');
    ghost?.remove();
    drag = null;
    document.body.classList.remove('grad-dragging');
    clearPaint();
    if (!active || !commit || !hit) return;

    if (hit.kind === 'swap') {
      emit('grad_swap', { a: id, b: hit.other });
      return;
    }
    if (hit.kind === 'column') {
      emit('grad_retile', { window: id, column: hit.column, slot: null, new_column: true });
      return;
    }
    // A drop onto its own position is not a move. Emitting it anyway would cost
    // a layout write and a full rebuild of the pane tree for no visible change.
    const here = positionOf(id);
    if (here && here.column === hit.column && (hit.slot === here.slot || hit.slot === here.slot + 1)) return;
    emit('grad_retile', { window: id, column: hit.column, slot: hit.slot, new_column: false });
  };

  document.addEventListener('mousedown', (event) => {
    if (event.button !== 0) return;
    const bar = event.target.closest?.('.grad-titlebar[data-window]');
    if (!bar) return;
    // The focus, restore and close buttons live inside the bar. A press on one
    // of them is that button's click, never the start of a drag.
    if (event.target.closest?.('.grad-winctl')) return;
    drag = { id: bar.dataset.window, bar, x: event.clientX, y: event.clientY, active: false, hit: null };
  });

  document.addEventListener('mousemove', (event) => {
    if (!drag) return;
    if (!drag.active) {
      // A press that never travels is a click -- the title bar's own focus
      // handler -- so the drag only starts once the pointer has committed to it.
      if (Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) < DRAG_THRESHOLD_PX) return;
      drag.active = true;
      document.body.classList.add('grad-dragging');
      drag.bar.classList.add('grad-drag-source');
      drag.ghost = document.createElement('div');
      drag.ghost.className = 'grad-drag-ghost';
      drag.ghost.textContent = drag.id;
      document.body.appendChild(drag.ghost);
    }
    drag.ghost.style.left = `${event.clientX + 14}px`;
    drag.ghost.style.top = `${event.clientY + 14}px`;
    drag.hit = hitTest(event.clientX, event.clientY, drag.id);
    paint(drag.hit);
    event.preventDefault();
  });

  document.addEventListener('mouseup', () => endDrag(true));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') endDrag(false);
  });

  /* ----------------------------------------------------------- shortcuts */
  document.addEventListener('keydown', (event) => {
    if (!event.altKey || event.ctrlKey || event.metaKey) return;
    const preset = { '1': 'tile', '2': 'stack', '3': 'full' }[event.key];
    if (!preset) return;
    emit('grad_preset', { preset });
    event.preventDefault();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'k' && (event.metaKey || event.ctrlKey)) {
      emit('grad_palette', {});
      event.preventDefault();
    }
  });

  /* ------------------------------------------------------ iframe overlay */
  /* An iframe registered here is positioned over its anchor element instead of
   * living inside it, so the pane tree can be torn down and rebuilt around it
   * without the browser reloading the document. */
  const frames = new Map();
  //: Assigned at the bottom of this block, where the loop it starts is defined.
  //: Only ever called from `gradRegisterFrame`, which runs long after.
  let startLoop = () => {};

  /* Placing an overlay costs a layout, and writing the four position properties
   * dirties layout again for the next reader. On the overwhelming majority of
   * passes the numbers are identical -- a pane that has not moved -- so the box
   * is remembered and the writes are skipped. This matters far more than it
   * looks: with a whole JupyterLab document in the frame, the layout each
   * needless write forces is not a small one. */
  const place = (frame, next) => {
    if (frame.__gradBox === next) return;
    frame.__gradBox = next;
    if (next === null) {
      frame.style.display = 'none';
      return;
    }
    frame.style.display = 'block';
    frame.style.left = `${next[0]}px`;
    frame.style.top = `${next[1]}px`;
    frame.style.width = `${next[2]}px`;
    frame.style.height = `${next[3]}px`;
  };

  const reflowFrames = () => {
    frames.forEach((frame, anchorId) => {
      const anchor = document.getElementById(anchorId);
      if (!anchor) {
        place(frame, null);
        return;
      }
      const box = anchor.getBoundingClientRect();
      const visible = box.width > 1 && box.height > 1 && box.bottom > 0 && box.top < window.innerHeight;
      if (!visible) {
        place(frame, null);
        return;
      }
      const next = [Math.round(box.left), Math.round(box.top), Math.round(box.width), Math.round(box.height)];
      const held = frame.__gradBox;
      if (held && held[0] === next[0] && held[1] === next[1] && held[2] === next[2] && held[3] === next[3]) return;
      place(frame, next);
    });
  };

  window.gradRegisterFrame = (anchorId, src, sandboxed) => {
    let frame = frames.get(anchorId);
    if (frame && frame.dataset.src === src) {
      reflowFrames();
      startLoop();
      return;
    }
    if (frame) frame.remove();
    frame = document.createElement('iframe');
    frame.className = 'grad-iframe-host';
    frame.dataset.src = src;
    // Notebook *output* is untrusted HTML from files that may have come from a
    // downloaded repository, so it is sandboxed. Lab is a server we started on
    // its own port behind a token we minted, and cannot function sandboxed.
    if (sandboxed) frame.setAttribute('sandbox', '');
    frame.setAttribute('allow', 'clipboard-read; clipboard-write');
    frame.src = src;
    document.body.appendChild(frame);
    frames.set(anchorId, frame);
    reflowFrames();
    startLoop();
  };

  window.gradDropFrame = (anchorId) => {
    const frame = frames.get(anchorId);
    if (frame) frame.remove();
    frames.delete(anchorId);
  };

  window.gradReflow = reflowFrames;

  /* ------------------------------------------------------- sticky transcript */
  /* A turn appends to the bottom of the transcript for as long as it runs, and
   * a reader watching it should not have to chase it with the scrollbar. But
   * scrolling up to re-read a tool's output has to survive the next token, so
   * this pins only while the reader is already at the bottom. Server-side this
   * would be a `run_javascript` per flush, fifteen times a second. */
  window.gradStickBottom = (id) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.gradStuck) return;
    el.dataset.gradStuck = '1';
    const SLACK_PX = 80;
    const atBottom = () => el.scrollHeight - el.scrollTop - el.clientHeight <= SLACK_PX;
    let pinned = true;
    el.addEventListener('scroll', () => { pinned = atBottom(); }, {passive: true});
    const stick = () => { if (pinned) el.scrollTop = el.scrollHeight; };
    new MutationObserver(stick).observe(el, {childList: true, subtree: true, characterData: true});
    stick();
  };

  window.addEventListener('resize', reflowFrames);
  window.addEventListener('scroll', reflowFrames, true);

  /* The pane tree is rebuilt by the server on every retile, so the anchor is a
   * different element each time; polling is more reliable than trying to
   * observe a node that keeps being replaced. What it does not have to be is
   * *per frame*. Everything a person does directly -- dragging a divider,
   * scrolling, resizing the window -- already reflows on its own event, so this
   * loop only has to catch changes the server made, and 15 Hz is well under the
   * rate anyone notices a frame settling into a new pane.
   *
   * The difference is not academic. The old loop measured and repositioned
   * sixty times a second for the lifetime of the page, whether or not anything
   * had moved and whether or not a frame existed at all -- and every one of
   * those passes reached into a layout containing an embedded JupyterLab. It
   * ran hardest in exactly the state it was most expensive in. */
  const POLL_MS = 66;
  let looping = false;
  let lastPoll = 0;

  const tick = (now) => {
    if (!frames.size) {
      // Nothing to place: stop entirely rather than idle. `gradRegisterFrame`
      // starts it again, so the workspace pays nothing for this until a window
      // that embeds something is actually open.
      looping = false;
      return;
    }
    if (now - lastPoll >= POLL_MS) {
      lastPoll = now;
      reflowFrames();
    }
    window.requestAnimationFrame(tick);
  };

  startLoop = () => {
    if (looping) return;
    looping = true;
    lastPoll = 0;
    window.requestAnimationFrame(tick);
  };
})();
