"""Run the browser-side remote-microphone protocol test (tests/remote_capture_smoke.js) with the Python suite."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_remote_capture_protocol_in_node():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js 未安裝，略過瀏覽器端遠端麥克風協定測試（tests/remote_capture_smoke.js）')
    result = subprocess.run([node, str(ROOT / 'tests' / 'remote_capture_smoke.js')], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ok' in result.stdout
