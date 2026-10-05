"""整體分析 job: professors A and B write independently, cross-review, professor C arbitrates."""
from __future__ import annotations

import re
from datetime import date

from ..jobs import Job, gather_all
from ..llm import ValidationError
from ..record.tags import strip_citations
from ..textutil import content_chars
from .common import NO_ANALYSIS_TEMPLATE, date_section, load_prompt, patient_section, render, retry_builder, section

ANALYSIS_PROMPT = 'prompt_analysis.txt'
CROSS_PROMPT = 'prompt_cross_review.txt'
ARBITRATION_PROMPT = 'prompt_arbitration.txt'
MARK_NOTES = '===ARBITRATION==='
MARK_FINAL = '===FINAL_AT==='
MIN_CHARS = 20
# The four headings prompt_cross_review.txt requires (matched by their leading words).
CROSS_HEADINGS = ('對方版本的優點', '對方版本的缺點', '與我方版本的分歧', '建議採納')

_NUMBER = r'(?:[（(]?(?:[一二三四五六七八九十]+|\d+)(?:[)）]|\s*[-.、．:：]|\s+))'   # "一-", "1.", "(二)", "三、"
_ORDINAL = re.compile('^' + _NUMBER + r'\s*')
_NUMBERED_LINE = re.compile(r'^(\s*)' + _NUMBER + r'\s*(\S.*)$')
_LEAD_MARKS = re.compile(r'^[\s#>*_`\-•·]+')


def non_empty(text: str) -> str:
    if len(text.strip()) < MIN_CHARS:
        raise ValidationError('輸出內容過短或空白。')
    return text.strip()


def _title_key(title: str) -> str:
    """The leading words of a section title: `處方(中藥或針灸)：` -> `處方`, `衛教/建議轉診` -> `衛教`."""
    return re.split(r'[/／(（:：]', title.strip(), maxsplit=1)[0].replace(' ', '').replace('　', '')


def template_sections(template: str) -> list[str]:
    """Top-level numbered item titles of the 分析模板 (empty when the template is free-form)."""
    found = [(len(m.group(1).expandtabs(4)), _title_key(m.group(2)))
             for m in (_NUMBERED_LINE.match(line) for line in template.splitlines()) if m]
    if not found:
        return []
    top = min(indent for indent, _ in found)
    keys: list[str] = []
    for indent, key in found:
        if indent == top and key and key not in keys:
            keys.append(key)
    return keys


def _heading(line: str) -> str:
    text = _LEAD_MARKS.sub('', line).replace('*', '').replace('_', '').strip()
    return _ORDINAL.sub('', text).replace(' ', '').replace('　', '')


def check_sections(text: str, keys: list[str] | tuple[str, ...], what: str) -> str:
    """Every key must open a heading line and have some content; raises ValidationError naming the problems."""
    if not keys:
        return text
    lines = text.splitlines()
    starts: dict[str, int] = {}
    for number, line in enumerate(lines):
        heading = _heading(line)
        for key in keys:
            if key not in starts and heading.startswith(key):
                starts[key] = number
                break
    missing = [key for key in keys if key not in starts]
    if missing:
        raise ValidationError(f'輸出缺少{what}：' + '、'.join(missing) + '。請逐項輸出，保留編號與標題。')
    order = sorted(starts.items(), key=lambda item: item[1])
    empty = []
    for position, (key, number) in enumerate(order):
        end = order[position + 1][1] if position + 1 < len(order) else len(lines)
        inline = re.split(r'[:：]', _heading(lines[number])[len(key):], maxsplit=1)
        body = (inline[1] if len(inline) > 1 else '') + ''.join(lines[number + 1:end])
        if not content_chars(body):
            empty.append(key)
    if empty:
        raise ValidationError(f'{what}沒有內容：' + '、'.join(empty) + '。每一項都要有內容（資訊不足請明寫「資訊不足」）。')
    return text


def parse_arbitration(text: str, keys: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    """Split professor C's output on the two marker lines; the final A&T must cover every template item."""
    lines = text.splitlines()
    try:
        notes_at = next(i for i, line in enumerate(lines) if line.strip() == MARK_NOTES)
        final_at = next(i for i, line in enumerate(lines) if line.strip() == MARK_FINAL)
    except StopIteration as exc:
        raise ValidationError(f'輸出缺少 {MARK_NOTES} 或 {MARK_FINAL} 標記行。') from exc
    if final_at < notes_at:
        raise ValidationError(f'{MARK_NOTES} 必須在 {MARK_FINAL} 之前。')
    notes = '\n'.join(lines[notes_at + 1:final_at]).strip()
    final = '\n'.join(lines[final_at + 1:]).strip()
    if len(final) < MIN_CHARS:
        raise ValidationError('最終 A&T 內容過短或空白。')
    check_sections(final, keys, '最終 A&T 的模板項目')
    return notes or '（模型未提供仲裁說明）', final


def case_block(patient: str, note: str, template: str, day: date) -> str:
    """The 【案例資料】 body every professor call starts from (stage 1 writers, cross reviews and the arbitrator)."""
    return '\n\n'.join([date_section(day), patient_section(patient),
                        section('今日病歷', strip_citations(note)),
                        section('分析模板', template.strip() or NO_ANALYSIS_TEMPLATE)])


async def run(session, job: Job):
    settings = session.settings
    names = {key: settings.professors[key].name for key in ('a', 'b', 'c')}
    styles = {key: settings.professors[key].role_style.strip() or '（無特別風格要求，請維持客觀、嚴謹、臨床導向。）'
              for key in ('a', 'b')}
    note_snapshot = session.note.get_current()
    case = case_block(session.patient_text, note_snapshot['note'], session.analysis_template, session.visit_date)
    keys = template_sections(session.analysis_template)

    def valid_at(text: str) -> str:
        return check_sections(non_empty(text), keys, '分析模板項目')

    def valid_review(text: str) -> str:
        return check_sections(non_empty(text), CROSS_HEADINGS, '評比標題')

    def valid_final(text: str):
        return parse_arbitration(text, keys)

    patient_version = session.patient_version
    session.log.emit('job_started', job_id=job.id, kind='analysis', base_index=session.note.current_index + 1,
                     base_snapshot_id=note_snapshot['id'], patient_version=patient_version, professors=names)
    analysis_prompt = load_prompt(ANALYSIS_PROMPT)
    cross_prompt = load_prompt(CROSS_PROMPT)
    arbitration_prompt = load_prompt(ARBITRATION_PROMPT)
    caller = session.caller

    # Stage 1: independent writing (parallel).
    job.set_stage(f'階段 1/3：{names["a"]}、{names["b"]} 獨立撰寫中')
    session.log.emit('analysis_stage', job_id=job.id, stage='1 獨立撰寫')

    def write(key: str, agent: str):
        system = render(analysis_prompt, name=names[key], role_style=styles[key])
        return caller.call(agent=agent, endpoint=settings.agents[agent], messages=retry_builder(system, case),
                           job_id=job.id, validator=valid_at)

    out_a, out_b = await gather_all(write('a', 'professor_a'), write('b', 'professor_b'))

    # Stage 2: cross review (parallel).
    job.set_stage(f'階段 2/3：{names["a"]}、{names["b"]} 互評中')
    session.log.emit('analysis_stage', job_id=job.id, stage='2 互評')

    def review(me: str, other: str, mine: str, theirs: str, agent: str):
        system = render(cross_prompt, name=names[me], other_name=names[other], role_style=styles[me])
        user = '\n\n'.join([section('案例資料', case), section('你的版本', mine),
                            section('對方的版本（' + names[other] + '）', theirs)])
        return caller.call(agent=agent, endpoint=settings.agents[agent], messages=retry_builder(system, user),
                           job_id=job.id, validator=valid_review)

    review_a_on_b, review_b_on_a = await gather_all(
        review('a', 'b', out_a.value, out_b.value, 'professor_a'),
        review('b', 'a', out_b.value, out_a.value, 'professor_b'))

    # Stage 3: arbitration.
    job.set_stage(f'階段 3/3：{names["c"]} 仲裁中')
    session.log.emit('analysis_stage', job_id=job.id, stage='3 仲裁')
    system = render(arbitration_prompt, name_a=names['a'], name_b=names['b'], name_c=names['c'])
    user = '\n\n'.join([section('案例資料', case),
                        section(f'{names["a"]} 的 A&T', out_a.value),
                        section(f'{names["b"]} 的 A&T', out_b.value),
                        section(f'{names["a"]} 對 {names["b"]} 的評比', review_a_on_b.value),
                        section(f'{names["b"]} 對 {names["a"]} 的評比', review_b_on_a.value)])
    out_c = await caller.call(agent='professor_c', endpoint=settings.agents['professor_c'],
                              messages=retry_builder(system, user), job_id=job.id, validator=valid_final)
    notes, final = out_c.value
    version = session.add_analysis(
        final_at=final, arbitration_notes=notes, note_snapshot_id=note_snapshot['id'], job_id=job.id,
        names=names,
        professors={'a': out_a.value, 'b': out_b.value, 'review_a_on_b': review_a_on_b.value,
                    'review_b_on_a': review_b_on_a.value, 'arbitration_raw': out_c.text},
        calls={'a': out_a.call_ids, 'b': out_b.call_ids, 'review_a_on_b': review_a_on_b.call_ids,
               'review_b_on_a': review_b_on_a.call_ids, 'c': out_c.call_ids},
        patient_version=patient_version)
    job.message = f'整體分析第 {version.index} 則完成。'
