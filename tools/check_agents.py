"""Run the real LLM agents end to end on a synthetic consultation (no microphone, no ASR model).

    .venv/Scripts/python.exe tools/check_agents.py [--keep] [--skip-analysis] [--no-stale-line] [--record-template PATH] [--url ... --model ...]

Uses config.json's endpoints unless overridden. Only synthetic text is sent. The scripted
"ASR" returns one dialogue line per audio window, so the transcript pipeline (OpenCC, rolling
LLM correction), record writer + reviewer, advice and the three-professor analysis all run
against the real model. Prints what each agent produced and every LLM call's outcome.
"""
import argparse
import asyncio
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

from mini.config import CONFIG_PATH, Settings                      # noqa: E402
from mini.llm import LLMClient, LLMScheduler                        # noqa: E402
from mini.visit import VisitSession                                 # noqa: E402
from mini.visit_store import VisitStore                             # noqa: E402
from tests.helpers import FakeASR, silent_source                    # noqa: E402

PATIENT = '測試患者，男，45歲。高血壓病史5年，規則服用降壓藥。上次就診(2026-09-20)主訴頭痛，診斷為緊張型頭痛，予川芎茶調散。'
STALE_LINE = '大便：一日三次，質稀'                      # no template-specific label: works with any record template
LABEL_RE = re.compile(r'^\s*[甲乙丙丁戊己庚辛壬癸](?:\.\d+)?\s*[-－.、]')
DIALOGUE = [
    '請坐，今天哪裡不舒服？', '医生，我这三天一直头痛，主要是两侧太阳穴胀痛。', '有沒有發燒或怕冷？',
    '没有发烧，但是有点怕风，流清鼻水。', '咳嗽嗎？', '偶尔干咳，没有痰。', '睡得怎麼樣？大便呢？',
    '睡眠还可以，大便偏干，两天一次。', '血壓有在量嗎？', '早上量过一百四十比九十，我有按时吃降压药。',
    '我看一下舌頭，舌淡紅，苔薄白。脈浮緊。', '没有胸闷胸痛，也没有过敏。',
]


def banner(text: str):
    print(f'\n{"=" * 78}\n{text}\n{"=" * 78}', flush=True)


async def run_job(session: VisitSession, kind: str) -> float:
    started = time.monotonic()
    job = session.start_job(kind)
    while not job.task.done():
        await asyncio.sleep(2)
        print(f'  … {job.label}：{job.stage}（{time.monotonic() - started:.0f}s）', flush=True)
    await job.task
    print(f'  → {job.label} {job.status}：{job.message}（{time.monotonic() - started:.1f}s）')
    return time.monotonic() - started


def llm_summary(session: VisitSession):
    banner('LLM 呼叫統計')
    for record in session.calls.values():
        status = f'ERROR: {record["error"][:70]}' if record['error'] else 'ok'
        print(f'  {record["call_id"]} {record["agent"]:<24} 第{record["attempt"]}次 {record["latency_ms"] or 0:>6} ms  {status}')
    transcript_errors = 0
    path = session.store.transcript_llm_path
    if path.exists():
        import json
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        transcript_errors = sum(1 for r in rows if r.get('error'))
        print(f'  逐字稿校稿：{len(rows)} 次呼叫，其中 {transcript_errors} 次失敗；平均 '
              f'{sum(r["latency_ms"] or 0 for r in rows) / max(len(rows), 1):.0f} ms')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url'), parser.add_argument('--model')
    parser.add_argument('--keep', action='store_true', help='keep the generated visit folder')
    parser.add_argument('--skip-analysis', action='store_true')
    parser.add_argument('--record-template', default='templates/record_template.txt',
                        help='record template to use (default: the working copy; templates/defaults/record_template.txt is the numbered original)')
    parser.add_argument('--no-stale-line', action='store_true',
                        help='do not seed an out-of-date physician line before the record job')
    args = parser.parse_args()
    settings = Settings.load(CONFIG_PATH) if CONFIG_PATH.exists() else Settings()
    for endpoint in settings.agents.values():
        if args.url:
            endpoint.api_url = args.url
        if args.model:
            endpoint.model_name = args.model
    settings.asr.overlap_seconds = 0.0            # one dialogue line per window, nothing to de-duplicate
    root = Path(tempfile.mkdtemp(prefix='mini-check-'))
    settings.visits_dir = str(root)
    settings.validate()
    print(f'端點：{settings.agents["record_writer"].api_url}  模型：{settings.agents["record_writer"].model_name}')
    print(f'審查 n={settings.review.pass_required_n}，最大輪數 {settings.review.max_review_rounds}；輸出：{root}')

    record_template = Path(args.record_template).read_text(encoding='utf-8')
    template_numbered = any(LABEL_RE.match(line) for line in record_template.splitlines())
    print(f'病歷模板：{args.record_template}（模板{"有" if template_numbered else "沒有"}甲乙丙…編號）')
    asr = FakeASR({i: line for i, line in enumerate(DIALOGUE)})
    store = VisitStore.allocate(root)
    session = VisitSession(
        store=store, settings=settings, patient_text=PATIENT, record_template=record_template,
        analysis_template=Path('templates/analysis_template.txt').read_text(encoding='utf-8'),
        client=LLMClient(), scheduler=LLMScheduler(settings.llm.max_concurrency), asr=asr,
        source_factory=lambda path: silent_source(settings, 6.0 * len(DIALOGUE), recording_path=path))
    try:
        banner('逐字稿（簡體 ASR → OpenCC → LLM 校稿）')
        started = time.monotonic()
        await session.start()
        await session.pipeline.wait_finished(timeout=900)
        print(session.pipeline.transcript_txt())
        print(f'（{time.monotonic() - started:.0f}s；校稿修訂 {len(session.pipeline.revisions)} 次）')

        banner('病歷書寫（寫病歷 agent + 幻覺審查）')
        if not args.no_stale_line:
            # An old, physician-typed line that the consultation contradicts ("大便偏乾，兩天一次"): the writer may
            # update it from the newer evidence, but must keep [醫師手動] and add the new source tag.
            session.edit_note_manual(STALE_LINE)
            print(f'（先由醫師手動寫入舊資料：{STALE_LINE}）\n  → {session.note.current_note()}')
        await run_job(session, 'record')
        print('\n' + session.note.current_note())
        if not args.no_stale_line:
            manual = next((line for line in session.note.current_note().splitlines() if '[醫師手動]' in line), '')
            ok = '[醫師手動]' in manual and '[語音#' in manual and '一日三次' not in manual
            print(f'\n【手動行更新檢查】{"PASS" if ok else "CHECK"}：{manual.strip()}')
        lines = [line for line in session.note.current_note().splitlines() if line.strip()]
        invented = sum(1 for line in lines if LABEL_RE.match(line))
        verdict = 'PASS' if (template_numbered or invented == 0) else 'FAIL'
        print(f'【編號檢查】{verdict}：病歷共 {len(lines)} 行，以甲乙丙…編號開頭的有 {invented} 行'
              f'（模板{"本身有編號" if template_numbered else "沒有編號，不該有"}）')
        for event in session.log.events:
            if event['type'] == 'review_result':
                print(f'  [審查第 {event["round"]} 輪] agree={event["agree"]} {event["comment"][:120]}')

        banner('問診建議')
        await run_job(session, 'advice')
        if session.advice:
            v = session.advice[-1]
            print(f'--- 西醫鑑別 ---\n{v.western_ddx}\n--- 中醫證型 ---\n{v.tcm_ddx}\n--- 建議問診 ---\n{v.next_questions}')

        if not args.skip_analysis:
            banner('整體分析（甲／乙獨立 → 互評 → 丙仲裁）')
            await run_job(session, 'analysis')
            if session.analysis:
                v = session.analysis[-1]
                print(f'--- 仲裁說明 ---\n{v.arbitration_notes}\n--- 最終 A&T ---\n{v.final_at}')
        llm_summary(session)
        folder = await session.finish()
        print(f'\n已存檔：{folder}')
    finally:
        if not args.keep:
            await asyncio.sleep(0.2)
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(main())
