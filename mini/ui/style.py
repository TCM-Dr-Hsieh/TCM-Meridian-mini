"""CSS and the small client-side helpers (copy, transcript auto-scroll)."""

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
'''
