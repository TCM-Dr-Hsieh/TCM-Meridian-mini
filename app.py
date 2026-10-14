"""TCM-Meridian-mini entry point.

    .venv/Scripts/python.exe app.py --open

Listens on 0.0.0.0:5050 (all network interfaces) so other computers on the LAN, or a tunnel such as cloudflared,
can reach it. MINI_HOST / MINI_PORT override (MINI_HOST=127.0.0.1 = this computer only). There is NO login: see
README.md before exposing it beyond a network you trust.
"""
import sys

from nicegui import app as ng_app
from nicegui import ui

from mini.config import server_binding
from mini.state import AppState
from mini.ui.page import REMOTE_JS, REMOTE_JS_URL, REPLAY_JS, REPLAY_JS_URL, MainPage
from mini.voice.remote import register_routes
from mini.voice.replay import register_replay_routes

state = AppState()

# Remote microphone: the browser-side capture script and the WebSocket the audio arrives on (same origin as the page).
ng_app.add_static_file(local_file=REMOTE_JS, url_path=REMOTE_JS_URL, max_cache_age=0)
register_routes(ng_app)

# Replay: click #n in the transcript to hear that part again. The player script and the route that cuts the clip out of
# the recording in progress (same origin as the page; only the current visit's audio, by time).
ng_app.add_static_file(local_file=REPLAY_JS, url_path=REPLAY_JS_URL, max_cache_age=0)
register_replay_routes(ng_app, state)


@ui.page('/', reconnect_timeout=60)
def index():
    MainPage(state)


ng_app.on_shutdown(state.shutdown)

if __name__ in {'__main__', '__mp_main__'}:
    host, port = server_binding()
    ui.run(host=host, port=port, title='TCM-Meridian-mini', language='zh-TW', favicon='🌿', reload=False,
           show='--open' in sys.argv)
