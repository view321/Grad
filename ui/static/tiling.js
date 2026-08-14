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
 *   3. Modifier chords. Alt+1/2/3 and Alt+drag are keyboard state the server
 *      never sees.
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

  /* ------------------------------------------------- alt+drag to retile */
  let dragging = null;

  document.addEventListener('mousedown', (event) => {
    if (!event.altKey) return;
    const bar = event.target.closest?.('.grad-titlebar[data-window]');
    if (!bar) return;
    dragging = bar.dataset.window;
    document.body.style.cursor = 'grabbing';
    event.preventDefault();
  });

  document.addEventListener('mouseup', (event) => {
    if (!dragging) return;
    const window_id = dragging;
    dragging = null;
    document.body.style.cursor = '';
    const column = event.target.closest?.('.grad-column');
    const tiles = document.querySelector('.grad-tiles');
    if (!tiles) return;
    const columns = Array.from(tiles.querySelectorAll(':scope > .grad-column'));
    // Past the right edge means "make a new column", which is why the index can
    // legitimately equal the column count.
    const index = column ? columns.indexOf(column) : columns.length;
    emit('grad_retile', { window: window_id, column: index });
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
