"""CSS and the small client-side helpers (copy, transcript auto-scroll, draggable floating panels)."""

CSS = '''
html, body {height:100%;}
body {background:#eef2f0; color:#1f2d27; font-family:"Segoe UI","Microsoft JhengHei",sans-serif;}
.nicegui-content {padding:0 !important; height:100vh; display:block;}
.shell {height:100vh; display:flex; flex-direction:column; gap:8px; padding:10px 14px; box-sizing:border-box; width:100%;}
.toolbar {display:flex; flex-wrap:wrap; gap:8px; align-items:center; width:100%;}
.toolbar .spacer {flex:1 1 auto;}
.mic-select {min-width:190px; max-width:260px;}
.source-toggle {border:1px solid #2d6a4f; border-radius:6px; overflow:hidden;}
.status-bar {display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center; width:100%; font-size:13px;
             background:#fff; border:1px solid #d5dfda; border-radius:10px; padding:6px 12px;}
.status-bar .job {font-weight:600; color:#1b4332;}
.status-bar .job.failed {color:#b3261e;}
.cols {flex:1 1 0; min-height:0; display:grid; grid-template-columns:1.25fr 1fr 0.9fr; gap:10px; width:100%;}
.work-col {display:flex; flex-direction:column; gap:10px; min-height:0; min-width:0;}
.panel {background:#fff; border:1px solid #d5dfda; border-radius:12px; padding:8px 12px; display:flex;
        flex-direction:column; min-height:0; min-width:0; box-shadow:0 1px 6px #1b433210;}
.g3 {flex:3 1 0;} .g2 {flex:2 1 0;} .g1 {flex:1 1 0;}
.panel-head {display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-bottom:4px;}
.panel-title {font-weight:700; color:#1b4332; font-size:15px;}
.panel-body {flex:1 1 0; min-height:0; overflow:auto; line-height:1.75; font-size:14px; overflow-wrap:anywhere;}
.panel-body .q-markdown {font-size:14px;}
.hint {color:#7a8a83; font-size:12px;}
.empty {color:#9aa8a1;}
.seg {margin:2px 0; padding:2px 6px; border-radius:6px;}
.seg-no {color:#7a8a83; font-size:11px; margin-right:6px; font-family:Consolas,monospace;}
.who {font-weight:700; margin-right:2px;}
.who.doctor {color:#1b4332;} .who.other {color:#9a5a00;} .who.unknown {color:#7f8b86; font-weight:600;}
.who.background {color:#929e99; font-weight:600;} .bg-text {color:#929e99;}
.seg.unlocked {background:#fff6d6;}
.seg.gap {background:#fde8d4; color:#9a4b00;}
.md-h1 {font-size:18px; font-weight:700; color:#1b4332; margin:10px 0 4px;}
.md-h2 {font-size:16px; font-weight:700; color:#1b4332; margin:8px 0 4px;}
.md-h3 {font-size:15px; font-weight:700; color:#2d6a4f; margin:6px 0 3px;}
.diff-title {font-weight:700; color:#1b4332; margin-bottom:6px;}
.diff-box {font-family:Consolas,"Microsoft JhengHei",monospace; font-size:13px; line-height:1.6; white-space:pre-wrap;
           background:#fafafa; border:1px solid #e0e0e0; border-radius:8px; padding:10px;}
.diff-add {background:#d4edda; color:#155724;}
.diff-del {background:#f8d7da; color:#721c24; text-decoration:line-through;}
.diff-hunk {color:#888; font-style:italic;}
.diff-none {color:#888;}
.version-label {font-size:12px; color:#52665d;}
.editor {flex:1 1 0; min-height:0;}
.editor .q-field__inner, .editor .q-field__control, .editor .q-field__control-container {height:100%;}
.editor textarea {height:100% !important; resize:none; font-size:14px; line-height:1.7; font-family:"Microsoft JhengHei",monospace;}
.nicegui-markdown h1 {font-size:19px; line-height:1.4; font-weight:700; margin:10px 0 4px;}
.nicegui-markdown h2 {font-size:17px; line-height:1.4; font-weight:700; margin:10px 0 4px;}
.nicegui-markdown h3 {font-size:16px; line-height:1.4; font-weight:700; margin:8px 0 3px;}
.nicegui-markdown h4, .nicegui-markdown h5, .nicegui-markdown h6 {font-size:15px; line-height:1.4; font-weight:700; margin:6px 0 2px;}
.nicegui-markdown p {margin:4px 0;}
.nicegui-markdown table {border-collapse:collapse; margin:6px 0; display:block; overflow-x:auto; max-width:100%;}
.nicegui-markdown th, .nicegui-markdown td {border:1px solid #cfd8d3; padding:4px 8px; vertical-align:top; font-size:13px;}
.nicegui-markdown th {background:#eaf1ed;}
.nicegui-markdown code, .nicegui-markdown pre {white-space:pre-wrap; overflow-wrap:anywhere;}
.nicegui-markdown ul, .nicegui-markdown ol {margin:4px 0; padding-left:22px;}
.float-panel {position:fixed; right:24px; top:96px; width:460px; height:min(62vh, 560px); min-width:260px; min-height:140px;
              z-index:3000; display:flex; flex-direction:column; resize:both; overflow:hidden; box-sizing:border-box;
              background:#fff; border:1px solid #b9c9c1; border-radius:12px; box-shadow:0 8px 28px #1b433238;}
.float-panel.dragging {user-select:none;}
.float-head {display:flex; align-items:center; gap:6px; padding:5px 8px 5px 12px; background:#eaf1ed; cursor:move;
             touch-action:none; user-select:none; border-bottom:1px solid #d5dfda;}
.float-title {font-weight:700; color:#1b4332; font-size:15px;}
.float-body {flex:1 1 0; min-height:0; overflow:auto; padding:10px 14px;}
.patient-text {white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.75; font-size:14px; user-select:text; cursor:text;}
.patient-text.empty {color:#9aa8a1;}
@media (max-width: 860px) {.cols {grid-template-columns:1fr; overflow:auto;} .work-col {min-height:70vh;}}
'''

JS = '''
window.miniFollow = true;
document.addEventListener('scroll', function (event) {
  const el = event.target;
  if (el && el.dataset && el.dataset.miniScroll === '1') {
    window.miniFollow = (el.scrollHeight - el.scrollTop - el.clientHeight) < 40;
  }
}, true);
window.miniScrollToEnd = function (id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.dataset.miniScroll = '1';
  if (window.miniFollow) el.scrollTop = el.scrollHeight;
};
window.miniCopy = async function (text) {
  // navigator.clipboard only exists in a secure context (https or localhost). A plain-http LAN address such as
  // http://192.168.1.20:5050 is not one, so fall back to the legacy copy command there.
  if (window.isSecureContext && navigator.clipboard) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* try the fallback */ }
  }
  try {
    const box = document.createElement('textarea');
    box.value = text;
    box.setAttribute('readonly', '');
    box.style.position = 'fixed';
    box.style.opacity = '0';
    document.body.appendChild(box);
    box.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(box);
    return ok;
  } catch (e) { return false; }
};
// Floating panels (class float-panel, header float-head): no backdrop, so the page behind stays usable. The header drags
// the panel (mouse or touch), the bottom-right corner resizes it (CSS), a double click on the header puts it back, and the
// position and size are remembered in this browser. A panel can never be dragged out of reach: its left and top edges stay
// inside the window, the whole header stays above the bottom edge (HEAD_H is only the fallback while the header is not
// laid out), and at least MIN_VISIBLE px of the title end stay inside on the right while dragging. When the panel opens
// and when the browser window shrinks, it is sized to the window and pulled fully into view, on both axes, where it fits.
window.miniFloat = (function () {
  const KEY = 'miniFloat:', MIN_VISIBLE = 80, HEAD_H = 40, DEFAULT_W = 460;
  let drag = null;
  function headOf(el) { const head = el.querySelector('.float-head'); return (head && head.offsetHeight) || HEAD_H; }
  function place(el, left, top) {
    // Left edge never past the window's left edge: the buttons sit at the right end of the header and are not draggable,
    // so a panel pushed left would leave nothing to grab. On the right, the title end (the draggable part) stays in.
    const x = Math.min(Math.max(left, 0), window.innerWidth - MIN_VISIBLE);
    const y = Math.min(Math.max(top, 0), window.innerHeight - headOf(el));
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    el.style.right = 'auto';
  }
  function save(el) {
    try {
      const r = el.getBoundingClientRect();
      localStorage.setItem(KEY + el.id, JSON.stringify({left: r.left, top: r.top, width: r.width, height: r.height}));
    } catch (e) { /* blocked storage: show() reads this page's own place and size back from the element; a reload or a double click on the header resets */ }
  }
  function load(id) {
    try { return JSON.parse(localStorage.getItem(KEY + id) || 'null'); } catch (e) { return null; }
  }
  function visible(el) { return el.getClientRects().length > 0; }
  function show(id) {
    const el = document.getElementById(id);
    if (!el) return;
    let s = load(id);
    if (!(s && s.width > 0) && el.style.left) {    // storage blocked: keep where this page had it, but still fit it below
      s = {left: parseFloat(el.style.left), top: parseFloat(el.style.top),
           width: parseFloat(el.style.width) || DEFAULT_W, height: parseFloat(el.style.height) || 400};
    }
    if (!(s && s.width > 0)) {
      s = {width: DEFAULT_W, height: Math.min(window.innerHeight * 0.62, 560),
           left: window.innerWidth - DEFAULT_W - 24, top: 96};
    }
    // sized to the window, then pulled fully into view where it fits (the saved place may come from a bigger window)
    const w = Math.min(s.width, window.innerWidth - 20), h = Math.min(s.height, window.innerHeight - 20);
    el.style.width = w + 'px';
    el.style.height = h + 'px';
    place(el, Math.min(s.left, window.innerWidth - w), Math.min(s.top, window.innerHeight - h));
  }
  function reset(el) {
    try { localStorage.removeItem(KEY + el.id); } catch (e) { /* nothing stored */ }
    el.style.left = el.style.top = el.style.width = el.style.height = '';
    el.style.right = '';
    show(el.id);
  }
  document.addEventListener('pointerdown', function (event) {
    const head = event.target.closest ? event.target.closest('.float-head') : null;
    if (!head || event.target.closest('button') || (event.button !== undefined && event.button !== 0)) return;
    const el = head.closest('.float-panel');
    const r = el.getBoundingClientRect();
    drag = {el: el, dx: event.clientX - r.left, dy: event.clientY - r.top};
    el.classList.add('dragging');
    try { head.setPointerCapture(event.pointerId); } catch (e) { /* the document-level listeners still follow the pointer */ }
    event.preventDefault();
  });
  document.addEventListener('pointermove', function (event) {
    if (drag) place(drag.el, event.clientX - drag.dx, event.clientY - drag.dy);
  });
  function release() {
    if (drag) drag.el.classList.remove('dragging');
    drag = null;
    // a drag or a resize (the CSS corner handle ends on a pointerup too): remember where every open panel is
    document.querySelectorAll('.float-panel').forEach(function (el) { if (visible(el)) save(el); });
  }
  document.addEventListener('pointerup', release);
  document.addEventListener('pointercancel', release);
  document.addEventListener('dblclick', function (event) {
    const head = event.target.closest ? event.target.closest('.float-head') : null;
    if (head && !event.target.closest('button')) reset(head.closest('.float-panel'));
  });
  window.addEventListener('resize', function () {
    document.querySelectorAll('.float-panel').forEach(function (el) {
      if (!visible(el)) return;
      // a smaller window must not leave the header's buttons or the content outside it
      if (el.offsetWidth > window.innerWidth - 20) el.style.width = (window.innerWidth - 20) + 'px';
      if (el.offsetHeight > window.innerHeight - 20) el.style.height = (window.innerHeight - 20) + 'px';
      const r = el.getBoundingClientRect();       // back fully into view, on both axes, where it fits
      place(el, Math.min(r.left, window.innerWidth - el.offsetWidth), Math.min(r.top, window.innerHeight - el.offsetHeight));
    });
  });
  return {show: show, reset: reset};
})();
'''
