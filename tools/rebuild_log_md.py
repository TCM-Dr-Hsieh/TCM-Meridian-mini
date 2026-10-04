"""Rebuild log.md from log.jsonl (e.g. for a visit whose app crashed before finishing).

    .venv/Scripts/python.exe tools/rebuild_log_md.py visits/2026-10-04-001
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

from mini.visit_store import render_log_md  # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    log = folder / 'log.jsonl'
    if not log.is_file():
        raise SystemExit(f'找不到 {log}')
    events = []
    for number, line in enumerate(log.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():                    # blank separator written after a torn line
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            print(f'略過第 {number} 行（可能是當機時寫到一半）。')
    (folder / 'log.md').write_text(render_log_md(folder.name, events), encoding='utf-8')
    print(f'已重建 {folder / "log.md"}（{len(events)} 筆事件）')


if __name__ == '__main__':
    main()
