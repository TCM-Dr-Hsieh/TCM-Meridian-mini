"""Edit config.json from the command line (used by setup.ps1; also handy for scripted installs).

    .venv/Scripts/python.exe tools/configure.py --llm-url http://host:8080/v1 --llm-model my-model [--llm-key KEY]
    .venv/Scripts/python.exe tools/configure.py --clear-asr-python

LLM options apply to all seven LLM interfaces (they can still be changed individually in the app).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from mini.config import CONFIG_PATH, Settings   # noqa: E402


def apply(settings: Settings, *, llm_url: str = '', llm_model: str = '', llm_key: str | None = None,
          asr_python: str | None = None) -> list[str]:
    """Apply the given changes in place; returns a human-readable list of what changed."""
    changes: list[str] = []
    for endpoint in settings.agents.values():
        if llm_url:
            endpoint.api_url = llm_url
        if llm_model:
            endpoint.model_name = llm_model
        if llm_key is not None:
            endpoint.api_key = llm_key
    if llm_url:
        changes.append(f'LLM URL = {llm_url}')
    if llm_model:
        changes.append(f'LLM 模型 = {llm_model}')
    if llm_key is not None:
        changes.append('LLM API Key 已更新')
    if asr_python is not None:
        settings.asr.python_path = asr_python
        changes.append(f'ASR Python 路徑 = {asr_python or "（專案內 .venv-asr）"}')
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--llm-url', default='')
    parser.add_argument('--llm-model', default='')
    parser.add_argument('--llm-key', default=None)
    parser.add_argument('--asr-python', default=None, help='path of the ASR worker Python')
    parser.add_argument('--clear-asr-python', action='store_true', help='use the project .venv-asr again')
    args = parser.parse_args()
    settings = Settings.load(CONFIG_PATH)
    asr_python = '' if args.clear_asr_python else args.asr_python
    changes = apply(settings, llm_url=args.llm_url, llm_model=args.llm_model, llm_key=args.llm_key,
                    asr_python=asr_python)
    settings.save(CONFIG_PATH)
    print('config.json 已更新：' + ('；'.join(changes) if changes else '（沒有變更）'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
