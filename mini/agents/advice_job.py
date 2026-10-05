"""問診建議 job: one call -> western DDx, TCM syndrome DDx, suggested questions."""
from __future__ import annotations

from ..jobs import Job
from ..llm import ValidationError
from ..record.tags import strip_citations
from ..textutil import extract_json
from .common import date_section, load_prompt, patient_section, retry_builder, section

PROMPT = 'prompt_ai_advice.txt'
FIELDS = ('western_ddx', 'tcm_ddx', 'next_questions')


def _as_text(value) -> str:
    if isinstance(value, list):
        return '\n'.join(f'- {item}' if not str(item).lstrip().startswith(('-', '*', '•')) else str(item)
                         for item in value)
    return str(value)


def parse_advice(text: str) -> dict:
    try:
        data = extract_json(text)
    except ValueError as exc:
        raise ValidationError(f'輸出不是合法 JSON：{exc}') from exc
    if not isinstance(data, dict):
        raise ValidationError('輸出必須是 JSON 物件。')
    missing = [name for name in FIELDS if not isinstance(data.get(name), (str, list)) or not data.get(name)]
    if missing:
        raise ValidationError('缺少或為空的欄位：' + '、'.join(missing))
    return {name: _as_text(data[name]).strip() for name in FIELDS}


async def run(session, job: Job):
    note_snapshot = session.note.get_current()
    patient_version = session.patient_version
    session.log.emit('job_started', job_id=job.id, kind='advice', base_index=session.note.current_index + 1,
                     base_snapshot_id=note_snapshot['id'], patient_version=patient_version)
    job.set_stage('問診建議產生中')
    user = '\n\n'.join([date_section(session.visit_date), patient_section(session.patient_text),
                        section('今日病歷', strip_citations(note_snapshot['note']))])
    outcome = await session.caller.call(agent='ai_advice', endpoint=session.settings.agents['ai_advice'],
                                        messages=retry_builder(load_prompt(PROMPT), user), job_id=job.id,
                                        validator=parse_advice)
    version = session.add_advice(outcome.value, note_snapshot['id'], outcome.call_ids, job.id, patient_version)
    job.message = f'問診建議第 {version.index} 則完成。'
