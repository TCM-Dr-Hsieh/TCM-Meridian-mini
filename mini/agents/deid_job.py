"""LLM 去識別化 job: today's date + patient data + today's record -> a de-identified copy of all three."""
from __future__ import annotations

import re
import unicodedata

from ..jobs import Job
from ..llm import ValidationError
from ..record.tags import strip_citations
from ..textutil import estimate_tokens
from .common import date_section, load_prompt, patient_section, retry_builder, section
from .deid_check import residual_warnings

PROMPT = 'prompt_de_identification.txt'
MARK_DATE, MARK_PATIENT, MARK_NOTE, MARK_SUMMARY = '===DATE===', '===PATIENT===', '===NOTE===', '===SUMMARY==='
MARKS = (MARK_DATE, MARK_PATIENT, MARK_NOTE, MARK_SUMMARY)
REQUIRED = (MARK_DATE, MARK_PATIENT, MARK_NOTE)           # 處理說明 is for the physician; a reply without it is still usable
D_DAY = 'D日'                                             # the only thing the date block may say
EXTRA_TITLE = '醫師額外指示（本次）'
MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS = 2048, 32_000


def _unfenced(text: str) -> str:
    """Models sometimes wrap a whole plain-text answer in a code fence."""
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith('```') and lines[-1].strip() == '```':
        return '\n'.join(lines[1:-1])
    return text


def _is_d_day(block: str) -> bool:
    """「D日」 up to spacing, full-width letters and a closing full stop; anything else (a real date, 「D日（星期二）」) is not."""
    return re.sub(r'[\s。.]+', '', unicodedata.normalize('NFKC', block)).upper() == D_DAY


def parse_deid(text: str) -> dict:
    """Split the reply on its marker lines into date / patient / note / summary; raises ValidationError when unusable."""
    lines = _unfenced(text).splitlines()
    at: dict[str, int] = {}
    for number, line in enumerate(lines):
        mark = line.strip()
        if mark in MARKS and mark not in at:
            at[mark] = number
    missing = [mark for mark in REQUIRED if mark not in at]
    if missing:
        raise ValidationError('輸出缺少標記行：' + '、'.join(missing) + '。請依格式輸出，標記行必須獨立一行、文字完全一致。')
    order = sorted(at, key=at.get)
    if order != [mark for mark in MARKS if mark in at]:
        raise ValidationError('標記行的順序必須是 ' + '、'.join(MARKS) + '。')
    blocks = {}
    for position, mark in enumerate(order):
        end = at[order[position + 1]] if position + 1 < len(order) else len(lines)
        blocks[mark] = '\n'.join(lines[at[mark] + 1:end]).strip()
    empty = [mark for mark in REQUIRED if not blocks[mark]]
    if empty:
        raise ValidationError('這些區塊沒有內容：' + '、'.join(empty) + '。原文空白時請寫「（空白）」。')
    if not _is_d_day(blocks[MARK_DATE]):                  # the visit date is the first thing to hide: never kept as written
        raise ValidationError(f'{MARK_DATE} 區塊只能寫「{D_DAY}」，不可寫出實際日期、星期或其他文字。')
    return {'date': D_DAY, 'patient': blocks[MARK_PATIENT], 'note': blocks[MARK_NOTE],
            'summary': blocks.get(MARK_SUMMARY, '')}


def output_budget(user: str) -> int:
    """max_tokens when the interface leaves it at 0: the reply repeats the input (pessimistic estimate) plus a summary."""
    return max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, int(estimate_tokens(user) * 1.2) + 1500))


async def run(session, job: Job, extra: str = ''):
    note_snapshot = session.note.get_current()
    patient_version = session.patient_version
    extra = extra.strip()
    source_patient = strip_citations(session.patient_text)
    source_note = strip_citations(note_snapshot['note'])
    session.log.emit('job_started', job_id=job.id, kind='deidentify', base_index=session.note.current_index + 1,
                     base_snapshot_id=note_snapshot['id'], patient_version=patient_version,
                     has_extra_instruction=bool(extra))
    job.set_stage('去識別化中')
    parts = [date_section(session.visit_date), patient_section(source_patient), section('今日病歷', source_note)]
    if extra:
        parts.append(section(EXTRA_TITLE, extra))
    user = '\n\n'.join(parts)
    endpoint = session.settings.agents['deidentifier']
    outcome = await session.caller.call(
        agent='deidentifier', endpoint=endpoint, messages=retry_builder(load_prompt(PROMPT), user), job_id=job.id,
        validator=parse_deid, max_tokens=None if endpoint.max_tokens else output_budget(user))
    parsed = outcome.value
    warnings = residual_warnings(visit_date=session.visit_date, source_patient=source_patient,
                                 source_note=source_note, date_text=parsed['date'], patient_text=parsed['patient'],
                                 note_text=parsed['note'])
    version = session.add_deid(parsed, warnings, extra, note_snapshot['id'], outcome.call_ids, job.id, patient_version)
    job.message = f'去識別化第 {version.index} 則完成' + (f'（{len(warnings)} 項提醒）。' if warnings else '。')
