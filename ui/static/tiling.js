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
    document.body.style.cursor = vertical ? 'row-resize' : 'col-resize';

    const onMove = (moveEvent) => {
      const delta = (vertical ? moveEvent.clientY : moveEvent.clientX) - startPos;
      const nextBefore = startBefore + delta;
      const nextAfter = startAfter - delta;
      if (nextBefore < MIN_PANE_PX || nextAfter < MIN_PANE_PX) return;
      before.style.setProperty('--grad-fraction', (nextBefore / totalPx).toFixed(6));
      after.style.setProperty('--grad-fraction', (nextAfter / totalPx).toFixed(6));
      reflowFrames();
    };

    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      emit('grad_resize', {
        axis: vertical ? 'slots' : 'columns',
        column: vertical ? Number(parent.dataset.columnIndex || 0) : null,
        fractions: fractionsOf(panes, panes.map((p) => size(p)), totalPx),
        total_px: Math.round(totalPx),
      });
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    event.preventDefault();
  };

  document.addEventListener('mousedown', (event) => {
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

  const reflowFrames = () => {
    frames.forEach((frame, anchorId) => {
      const anchor = document.getElementById(anchorId);
      if (!anchor) {
        frame.style.display = 'none';
        return;
      }
      const box = anchor.getBoundingClientRect();
      const visible = box.width > 1 && box.height > 1 && box.bottom > 0 && box.top < window.innerHeight;
      frame.style.display = visible ? 'block' : 'none';
      if (!visible) return;
      frame.style.left = `${Math.round(box.left)}px`;
      frame.style.top = `${Math.round(box.top)}px`;
      frame.style.width = `${Math.round(box.width)}px`;
      frame.style.height = `${Math.round(box.height)}px`;
    });
  };

  window.gradRegisterFrame = (anchorId, src, sandboxed) => {
    let frame = frames.get(anchorId);
    if (frame && frame.dataset.src === src) {
      reflowFrames();
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
  };

  window.gradDropFrame = (anchorId) => {
    const frame = frames.get(anchorId);
    if (frame) frame.remove();
    frames.delete(anchorId);
  };

  window.gradReflow = reflowFrames;

  window.addEventListener('resize', reflowFrames);
  window.addEventListener('scroll', reflowFrames, true);
  // The pane tree is rebuilt by the server on every retile, so the anchor is a
  // different element each time; polling one frame per animation frame is both
  // cheaper and more reliable than trying to observe a node that keeps being
  // replaced.
  const tick = () => {
    reflowFrames();
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
})();
