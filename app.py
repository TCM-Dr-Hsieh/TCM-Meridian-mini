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
from mini.ui.page import MainPage

state = AppState()


@ui.page('/', reconnect_timeout=60)
def index():
    MainPage(state)


ng_app.on_shutdown(state.shutdown)

if __name__ in {'__main__', '__mp_main__'}:
    host, port = server_binding()
    ui.run(host=host, port=port, title='TCM-Meridian-mini', language='zh-TW', favicon='🌿', reload=False,
           show='--open' in sys.argv)
