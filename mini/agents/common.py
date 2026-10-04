"""Prompt loading and shared message-building helpers for the agents."""
from __future__ import annotations

from ..config import PROMPTS_DIR

NO_ANALYSIS_TEMPLATE = '（無分析模板，請依常規臨床思路自由組織。）'
NO_RECORD_TEMPLATE = '（無病歷模板，請依常規中醫門診病歷格式自行組織。）'

_cache: dict[str, str] = {}


def load_prompt(filename: str, *, cache: bool = True) -> str:
    if cache and filename in _cache:
        return _cache[filename]
    text = (PROMPTS_DIR / filename).read_text(encoding='utf-8')
    if cache:
        _cache[filename] = text
    return text


# Blocks both the writer and the reviewer must read identically (as in the original project, where the same text is
# repeated in both prompts); kept in one file each so the two can never drift apart.
SHARED_BLOCKS = {'clinical_status': 'shared_clinical_status.txt',
                 'hallucination_categories': 'shared_hallucination_categories.txt'}


def load_system_prompt(filename: str) -> str:
    """A prompt file with its `{clinical_status}` / `{hallucination_categories}` placeholders filled in."""
    return render(load_prompt(filename), **{key: load_prompt(name).strip() for key, name in SHARED_BLOCKS.items()})


def render(template: str, **values: str) -> str:
    """Replace {key} placeholders literally (prompts contain JSON braces, so no str.format)."""
    for key, value in values.items():
        template = template.replace('{' + key + '}', value)
    return template


def section(title: str, body: str) -> str:
    return f'## 【{title}】\n{body.strip(chr(10)) if body.strip() else "（空白）"}'


def patient_section(text: str) -> str:
    return section('患者匯入資料（歷史資料，來源標籤 [歷史]）', text)


def messages(system: str, user: str) -> list[dict]:
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def retry_builder(system: str, user: str):
    """messages callable for LLMCaller: tells the model why the previous attempt was unusable."""
    def build(last_error: str | None) -> list[dict]:
        if not last_error:
            return messages(system, user)
        reason = section('上一次輸出無效的原因（請修正後重新輸出）', last_error)
        return messages(system, user + '\n\n' + reason)
    return build
