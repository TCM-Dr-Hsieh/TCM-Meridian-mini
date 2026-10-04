"""Development aid: a tiny OpenAI-compatible server with canned, role-aware replies.

    .venv\\Scripts\\python.exe tools\\fake_llm_server.py [--port 8099] [--delay 0.5] [--fail-reviews N]

Point every interface at http://127.0.0.1:8099/v1 (model name: fake) to exercise the whole UI
without a real model. Replies are valid for the app's contracts (line operations with real
segment numbers, reviewer JSON, advice JSON, arbitration markers).
"""
import argparse
import asyncio
import json
import re

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

ROLE_MARKERS = [
    ('corrector', '三段滾動校字員'),
    ('reviewer', '你是一位嚴謹且嚴格的「幻覺與過度聲稱審查員」'),
    ('writer', '你是「病歷書寫助理」'),
    ('advice', '問診建議助理'),
    ('arbitration', '擔任仲裁者'),
    ('cross', '評比對方的版本'),
    ('analysis', '獨立撰寫「分析與處置」'),
]
ARGS = argparse.Namespace(delay=0.3, fail_reviews=0)
STATE = {'reviews': 0, 'writes': 0}


def role_of(system: str) -> str:
    for role, marker in ROLE_MARKERS:
        if marker in system:
            return role
    return 'unknown'


def reply(role: str, system: str, user: str) -> str:
    if role == 'corrector':
        payload = json.loads(user)
        return json.dumps({'segments': [{'index': r['index'], 'text': r['current']} for r in payload['segments']]},
                          ensure_ascii=False)
    if role == 'writer':
        STATE['writes'] += 1
        numbers = [int(n) for n in re.findall(r'\[#(\d+) ', user)]
        first, last = (numbers[0], numbers[-1]) if numbers else (1, 1)
        note_part = user.split('## 【今日病歷（附行號）】', 1)[1].split('## 【病歷模板】', 1)[0] if '今日病歷（附行號）' in user else ''
        existing = [text for text in note_part.splitlines() if re.match(r'\s*\d+ \| ', text)]
        if not existing:
            ops = [{'op': 'insert', 'line': 1, 'content': f'甲- 現病史：患者近期陳述如逐字稿所述[語音#{first}]'},
                   {'op': 'insert', 'line': 1, 'content': '乙- 過去病史：高血壓[歷史]'},
                   {'op': 'insert', 'line': 1, 'content': '辛- 問診：其餘項目未知（尚未詢問）'}]
        else:
            ops = [{'op': 'insert', 'line': len(existing) + 1,
                    'content': f'辛.2- 全身症狀：逐字稿後段另有補充[語音#{last}]'}]
        return json.dumps({'thinking': 'fake', 'operations': ops, 'summary': f'假模型第 {STATE["writes"]} 次書寫'},
                          ensure_ascii=False)
    if role == 'reviewer':
        STATE['reviews'] += 1
        if STATE['reviews'] <= ARGS.fail_reviews:
            return json.dumps({'pass': False, 'issues': [{'line': 1, 'category': 'G-2', 'quote': '甲- 現病史',
                                                           'problem': '（假模型）示範退件', 'fix_hint': '重新核對'}],
                               'comment': '示範：第一輪退件'}, ensure_ascii=False)
        return json.dumps({'pass': True, 'issues': [], 'comment': '無須修改'}, ensure_ascii=False)
    if role == 'advice':
        return json.dumps({'western_ddx': '1. 上呼吸道感染（需排除肺炎）\n2. 過敏性鼻炎',
                           'tcm_ddx': '1. 風寒束表\n2. 風熱犯肺\n   - 鑑別：咽痛、痰色',
                           'next_questions': '1. 發燒幾天、最高幾度？（鑑別感染）\n2. 咳嗽有痰嗎？顏色？（鑑別寒熱）'},
                          ensure_ascii=False)
    name = re.search(r'你是 (\S+?)，', system)
    who = name.group(1) if name else '教授'
    if role == 'analysis':
        return (f'## 一- 西醫診斷\n- 上呼吸道感染（{who}版，待確認）\n## 二- 中醫診斷\n- 感冒\n'
                '## 三- 中醫病機/證型\n- 風寒束表（傾向）\n## 四- 中醫治則\n- 疏風散寒\n'
                '## 五- 處方(中藥或針灸)\n- 建議由醫師決定\n## 六- 衛教/建議轉診\n- 多休息、多喝水')
    if role == 'cross':
        return (f'## 對方版本的優點\n- 診斷方向合理（{who}評）\n## 對方版本的缺點與風險\n- 無明顯安全疑慮\n'
                '## 與我方版本的分歧\n- 實質一致\n## 建議採納與不建議採納\n- 建議採納')
    if role == 'arbitration':
        return ('===ARBITRATION===\n兩位教授意見大致一致；處方請醫師決定。\n===FINAL_AT===\n'
                '## 一- 西醫診斷\n- 上呼吸道感染（待確認）\n## 二- 中醫診斷\n- 感冒\n'
                '## 三- 中醫病機/證型\n- 風寒束表（傾向）\n## 四- 中醫治則\n- 疏風散寒\n'
                '## 五- 處方(中藥或針灸)\n- 建議由醫師決定\n'
                '## 六- 衛教/建議轉診\n- 多休息；若高燒不退請就醫')
    return '（假模型）'


async def models(request):
    return JSONResponse({'data': [{'id': 'fake'}]})


async def chat(request):
    body = await request.json()
    messages = body['messages']
    system, user = messages[0]['content'], messages[-1]['content']
    role = role_of(system)
    await asyncio.sleep(ARGS.delay)
    print(f'[fake-llm] {role} ({len(user)} chars)', flush=True)
    return JSONResponse({'choices': [{'message': {'content': reply(role, system, user)}, 'finish_reason': 'stop'}],
                         'usage': {'total_tokens': 1}})


app = Starlette(routes=[Route('/v1/models', models), Route('/v1/chat/completions', chat, methods=['POST'])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--delay', type=float, default=0.3)
    parser.add_argument('--fail-reviews', type=int, default=0, help='reject the first N reviews (demo of rewrite loop)')
    parsed = parser.parse_args()
    ARGS.delay, ARGS.fail_reviews = parsed.delay, parsed.fail_reviews
    uvicorn.run(app, host='127.0.0.1', port=parsed.port, log_level='warning')
