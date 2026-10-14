"""Run the browser-side replay-window test (tests/replay_smoke.js) with the Python suite."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_replay_window_in_node():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js 未安裝，略過回聽視窗的瀏覽器端測試（tests/replay_smoke.js）')
    result = subprocess.run([node, str(ROOT / 'tests' / 'replay_smoke.js')], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ok' in result.stdout
