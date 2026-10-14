"""Manual browser harness for the replay window (mini/ui/replay.js); not collected by pytest.

    .venv\\Scripts\\python.exe tests\\replay_harness.py        then open http://127.0.0.1:5099/

A page with fake transcript rows (the same markup MainPage.render_transcript writes), the real CSS / replay.js, and the
real clip route over a generated 70 s recording (the tone rises every 3 s, so by ear you can tell where a clip starts).
The page prints a live JSON probe of the window's state. Buttons for the cross-visit cases:

  * 「換成另一場看診（頁面與伺服器一起）」: the next visit starts and the page renders its rows with its token, with no
    empty state in between (a throttled background tab). The open player must close by itself.
  * 「只換伺服器的看診」: the server's visit changes but this page keeps the old rows; a click must be refused (409) and
    never return the new visit's audio.
"""
import json
import secrets
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

from mini.ui.style import CSS, JS  # noqa: E402
from mini.voice.replay import register_replay_routes  # noqa: E402

RATE = 16000
TEXTS = ['張先生。', '你好，請問今天是什麼原因？', '要來看診呢？', '醫師你好，我今', '天是腸胃不舒服', '，要來看診。']


def make_recording(path: Path):
    t = np.arange(70 * RATE) / RATE
    audio = (np.sin(2 * np.pi * (200 + 40 * (t // 3)) * t) * 0.2 * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())


def build_app(folder: Path) -> FastAPI:
    wav = folder / 'audio.wav'
    make_recording(wav)
    state = SimpleNamespace(visit=None)
    count = {'n': 0}

    def next_visit():
        count['n'] += 1
        state.visit = SimpleNamespace(visit_id=f'visit-{count["n"]}', replay_token=secrets.token_urlsafe(16),
                                      store=SimpleNamespace(audio_path=wav))
        return state.visit

    next_visit()
    app = FastAPI()
    register_replay_routes(app, state)

    @app.get('/mini-replay.js')
    def js():
        return FileResponse(ROOT / 'mini' / 'ui' / 'replay.js', media_type='text/javascript')

    @app.get('/rows')
    def rows():
        return {'token': state.visit.replay_token, 'visit': state.visit.visit_id}

    @app.get('/swap-server')
    def swap_server():
        return {'token': next_visit().replay_token}

    @app.get('/')
    def index():
        return HTMLResponse(PAGE.replace('__CSS__', CSS).replace('__JS__', JS).replace('__TEXTS__', json.dumps(TEXTS)))

    return app


PAGE = '''<!doctype html><html><head><meta charset="utf-8"><style>__CSS__</style>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<script>__JS__</script><script src="/mini-replay.js"></script></head>
<body style="background:#eef2f0;margin:0">
<div style="width:420px;height:300px;margin:20px;display:flex" class="panel"><div class="panel-body" id="tx"></div></div>
<div style="margin:20px">
  <button id="swap-both">換成另一場看診（頁面與伺服器一起）</button>
  <button id="swap-server">只換伺服器的看診</button>
</div>
<pre id="probe" style="margin:20px;font-size:13px;background:#fff;padding:8px;width:760px;white-space:pre-wrap"></pre>
<script>
const TEXTS = __TEXTS__;
function render(token) {
  document.getElementById('tx').innerHTML = TEXTS.map(function (text, i) {
    const start = (12 + 3 * i).toFixed(2), clock = '00:' + String(12 + 3 * i).padStart(2, '0');
    return '<div class="seg" data-idx="' + (i + 1) + '" data-start="' + start + '"><span class="seg-no" data-idx="' + (i + 1) +
           '" data-start="' + start + '" data-token="' + token + '">#' + (i + 1) + ' ' + clock + '</span>' + text + '</div>';
  }).join('');
}
fetch('/rows').then(function (r) { return r.json(); }).then(function (d) { render(d.token); });
document.getElementById('swap-both').onclick = function () {
  fetch('/swap-server').then(function (r) { return r.json(); }).then(function (d) { render(d.token); });
};
document.getElementById('swap-server').onclick = function () { fetch('/swap-server'); };
setInterval(function () {
  const q = function (s) { return document.querySelector(s); }, p = q('#mini-replay');
  q('#probe').textContent = JSON.stringify({
    open: !!p && getComputedStyle(p).display !== 'none',
    title: q('#mini-replay .float-title') && q('#mini-replay .float-title').textContent,
    time: q('.replay-time') && q('.replay-time').textContent,
    start: q('[data-testid=replay-start]') && q('[data-testid=replay-start]').value,
    msg: q('[data-testid=replay-msg]') && q('[data-testid=replay-msg]').textContent,
    barMax: q('[data-testid=replay-bar]') && q('[data-testid=replay-bar]').max,
    replaying: [].map.call(document.querySelectorAll('.seg.replaying'), function (r) { return r.dataset.idx; })
  });
}, 200);
</script></body></html>'''


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        uvicorn.run(build_app(Path(folder)), host='127.0.0.1', port=5099, log_level='warning')
