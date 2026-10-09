"""病歷書寫 job: writer agent (line operations) + hallucination reviewer, cumulative pass counting."""
from __future__ import annotations

from dataclasses import dataclass

from ..jobs import Job, JobFailed
from ..llm import ValidationError
from ..record.diff import build_diff_context
from ..record.line_ops import apply_operations, number_lines, parse_operations
from ..textutil import extract_json
from .common import (NO_RECORD_TEMPLATE, date_section, load_system_prompt, messages, patient_section, retry_builder,
                     section)

WRITER_PROMPT = 'prompt_update_record.txt'
REVIEWER_PROMPT = 'prompt_hallucination_corrector.txt'


@dataclass
class WriterOutput:
    candidate: str
    ops: list
    logs: list[str]
    summary: str
    thinking: str


@dataclass
class ReviewOutput:
    passed: bool
    thinking: str
    comment: str


def writer_user_prompt(patient: str, transcript: str, note: str, template: str,
                       rejected: list[tuple[str, 'ReviewOutput']], last_error: str | None, history: str = '',
                       today: str = '') -> str:
    """`note` is what the operations apply to: the job's base version on the first write, and on every rewrite the
    version the reviewer just rejected (as in the original project), so earlier fixes are never undone and the
    reviewer's line numbers match what the writer sees. `rejected` holds every rejected version with its review.
    `today` is the ready-made 【今日看診日期】 block (see `date_section`); it leads the prompt."""
    parts = [today] if today else []
    parts += [patient_section(patient), section('逐字稿', transcript),
             section('病歷修改 diff 過程', history),
             section('今日病歷（附行號）', number_lines(note)),
             section('病歷模板', template or NO_RECORD_TEMPLATE)]
    if rejected:
        parts.append(section('歷次審查意見與被退件內容', rejected_history(rejected)))
        parts.append('請根據審查員的歷次意見，修改上方【今日病歷（附行號）】（即你上一輪被退件的版本）。輸出你的修改操作。')
    if last_error:
        parts.append(section('上一次輸出無效的原因（請修正後重新輸出）', last_error))
    return '\n\n'.join(parts)


def rejected_history(rejected: list[tuple[str, 'ReviewOutput']]) -> str:
    """Every earlier rejected version and the reviewer's opinion on it, oldest first (original: 歷次修改建議與被退件內容)."""
    parts = []
    for number, (content, review) in enumerate(rejected, 1):
        parts.append(f'[第 {number} 次被退件的病歷內容]\n{content}\n\n'
                     f'[第 {number} 次審查員修改建議]\n{review.comment}')
    return '\n\n'.join(parts)


def reviewer_user_prompt(patient: str, transcript: str, candidate: str, history: str = '', today: str = '') -> str:
    """As in the original project: the whole proposed note plus the version history up to (not including) this
    update; the reviewer is not told which lines changed, and receives the proposed note without line numbers.
    `today` is the same 【今日看診日期】 block the writer received."""
    parts = [today] if today else []
    parts += [patient_section(patient), section('逐字稿', transcript), section('病歷修改 diff 過程', history),
              section('即將登載的今日病歷', candidate)]
    return '\n\n'.join(parts)


def parse_review(text: str) -> ReviewOutput:
    try:
        data = extract_json(text)
    except ValueError as exc:
        raise ValidationError(f'審查輸出不是合法 JSON：{exc}') from exc
    required = {'thinking', 'agree', 'comment'}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValidationError('審查輸出必須是包含 "thinking"、"agree"、"comment" 三個欄位的 JSON 物件。')
    thinking = data['thinking']
    agree = data['agree']
    comment = data['comment']
    if not isinstance(thinking, str) or not thinking.strip():
        raise ValidationError('"thinking" 必須是非空字串，先記錄兩階段檢查再下結論。')
    if not isinstance(agree, str) or agree.strip().lower() not in ('yes', 'no'):
        raise ValidationError('"agree" 必須是字串 "yes" 或 "no"。')
    if not isinstance(comment, str):
        raise ValidationError('"comment" 必須是字串。')
    agree = agree.strip().lower()
    comment = comment.strip()
    if agree == 'no' and not comment:
        raise ValidationError('agree="no" 時 "comment" 不可為空，必須指出問題與修正方向。')
    return ReviewOutput(agree == 'yes', thinking.strip(), comment)


async def run(session, job: Job):
    settings = session.settings
    base_snapshot = session.note.get_current()
    base = base_snapshot['note']
    snap = session.pipeline.snapshot()
    patient = session.patient_text
    template = session.record_template
    required = settings.review.pass_required_n
    max_rounds = settings.review.max_review_rounds
    patient_version = session.patient_version
    # How the current note came to be (initial version -> the version being edited), like the original project.
    history = build_diff_context(session.note.snapshots, session.note.current_index)
    today = date_section(session.visit_date)          # 【今日看診日期】 (西元 + 民國), given to the writer and the reviewer
    session.log.emit('job_started', job_id=job.id, kind='record', base_index=session.note.current_index + 1,
                     base_snapshot_id=base_snapshot['id'], patient_version=patient_version, transcript_upto=snap.upto,
                     transcript_max_index=snap.max_index, unlocked=snap.unlocked, review_required=required,
                     max_review_rounds=max_rounds,
                     **({'speaker_marks': snap.marked, 'speaker': session.speaker.role_summary()}
                        if session.speaker is not None else {}))
    writer_prompt = load_system_prompt(WRITER_PROMPT)
    reviewer_prompt = load_system_prompt(REVIEWER_PROMPT)
    call_ids: list[str] = []

    async def write(working: str, rejected: list[tuple[str, ReviewOutput]], attempt_no: int) -> WriterOutput:
        job.set_stage(f'撰寫中（第 {attempt_no} 次）')

        def build(last_error):
            return messages(writer_prompt, writer_user_prompt(patient, snap.text, working, template,
                                                              rejected, last_error, history, today))

        def validate(text: str) -> WriterOutput:
            try:
                data = extract_json(text)
                ops = parse_operations(data)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            ok, candidate, logs, failures = apply_operations(working, ops)
            if not ok:
                session.log.emit('ops_rejected', job_id=job.id, problems=failures)
                raise ValidationError('行級操作不合法，整批退回：' + '；'.join(failures))
            return WriterOutput(candidate, ops, logs, str(data.get('summary', '')), str(data.get('thinking', '')))

        outcome = await session.caller.call(agent='record_writer', endpoint=settings.agents['record_writer'],
                                            messages=build, job_id=job.id, validator=validate)
        call_ids.extend(outcome.call_ids)
        result: WriterOutput = outcome.value
        session.log.emit('writer_ops', job_id=job.id, summary=result.summary, logs=result.logs,
                         operations=result.ops, call_ids=outcome.call_ids,
                         rewrite_of_rejected=len(rejected))    # 0 = written from the base version
        return result

    async def review(candidate: str, round_no: int) -> ReviewOutput:
        job.set_stage(f'審查中（第 {round_no} 輪，已通過 {passes}/{required}）')
        user = reviewer_user_prompt(patient, snap.text, candidate, history, today)
        outcome = await session.caller.call(agent='hallucination_corrector',
                                            endpoint=settings.agents['hallucination_corrector'],
                                            messages=retry_builder(reviewer_prompt, user), job_id=job.id,
                                            validator=parse_review)
        call_ids.extend(outcome.call_ids)
        return outcome.value

    passes = rounds = writes = 0
    working = base                                    # what the next write's operations apply to
    rejected: list[tuple[str, ReviewOutput]] = []     # every rejected version with its review, oldest first
    last_review: ReviewOutput | None = None
    while True:
        writes += 1
        written = await write(working, rejected, writes)
        candidate = written.candidate
        if not rejected and not written.ops:
            # As in the original (Record_Subagent: `if not operations`), an EMPTY operation list skips the reviewer, so
            # this is NOT "checked for omissions". A valid operation that leaves the text unchanged is different: the
            # original still reviews it (see the identical-result handling below).
            job.message = '病歷無需更新（書寫助理沒有提出任何修改；這種情況不會送審查員，所以沒有經過審查員的遺漏檢查）。'
            return
        if required == 0:
            session.log.emit('review_skipped', job_id=job.id)
            if candidate == base:                     # original: an identical note is never pushed as a new version
                job.message = '病歷無需更新（操作套用後內容與目前版本相同；未審查）。'
                return
            meta = {'job_id': job.id, 'review': {'skipped': True, 'rounds': 0, 'passes': 0, 'required': 0},
                    'call_ids': call_ids, 'summary': written.summary, 'patient_version': patient_version}
            session.push_note(candidate, session.next_writer_source(skipped=True), meta)
            job.message = '病歷已更新（未審查）。'
            return
        while True:
            if rounds >= max_rounds:
                raise JobFailed(_review_failure(rounds, passes, required, last_review))
            rounds += 1
            last_review = await review(candidate, rounds)
            session.log.emit('review_result', job_id=job.id, round=rounds, passed=last_review.passed,
                             passes=passes + (1 if last_review.passed else 0), required=required,
                             thinking=last_review.thinking,
                             agree='yes' if last_review.passed else 'no', comment=last_review.comment,
                             **{'pass': last_review.passed})
            if last_review.passed:
                passes += 1
                if passes >= required:
                    if candidate == base:             # reviewed and fine, but there is nothing new to write
                        job.message = (f'審查通過（審查 {rounds} 輪，通過 {passes} 次），但內容與目前版本相同，'
                                       '未建立新版本。')
                        return
                    meta = {'job_id': job.id, 'review': {'skipped': False, 'rounds': rounds, 'passes': passes,
                                                         'required': required},
                            'call_ids': call_ids, 'summary': written.summary, 'patient_version': patient_version}
                    session.push_note(candidate, session.next_writer_source(), meta)
                    job.message = f'病歷已更新（審查 {rounds} 輪，通過 {passes} 次）。'
                    return
                continue                              # review the same candidate again
            rejected.append((candidate, last_review))
            if rounds >= max_rounds:
                raise JobFailed(_review_failure(rounds, passes, required, last_review))
            working = candidate                       # rewrite ON the rejected version, with all opinions so far
            break


def _review_failure(rounds: int, passes: int, required: int, review: ReviewOutput | None) -> str:
    text = f'審查未通過：已達最大審查輪數 {rounds}（累積通過 {passes}/{required}），病歷未寫入。'
    if review and review.comment:
        text += f' 最後一輪意見：{review.comment}'
    return text
