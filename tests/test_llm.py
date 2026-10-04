import asyncio

import httpx
import pytest

from mini.config import Endpoint
from mini.llm import (CallFailed, LLMCaller, LLMClient, LLMError, LLMScheduler, PRIORITY_BACKGROUND,
                      PRIORITY_USER, ValidationError)
from tests.helpers import FakeLLM

ENDPOINT = Endpoint(api_key='sekret')
MSG = [{'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'y'}]


# --- scheduler -------------------------------------------------------------
async def test_scheduler_enforces_the_concurrency_limit():
    scheduler = LLMScheduler(2)
    running = peak = 0

    async def work():
        nonlocal running, peak
        async with scheduler.slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1

    await asyncio.gather(*(work() for _ in range(6)))
    assert peak == 2 and scheduler.active == 0 and scheduler.queued == 0


async def test_scheduler_serves_user_priority_before_background_waiters():
    scheduler = LLMScheduler(1)
    order: list[str] = []
    gate = asyncio.Event()

    async def holder():
        async with scheduler.slot():
            await gate.wait()

    async def waiter(name, priority):
        async with scheduler.slot(priority):
            order.append(name)

    first = asyncio.create_task(holder())
    await asyncio.sleep(0)
    tasks = [asyncio.create_task(waiter('bg1', PRIORITY_BACKGROUND)),
             asyncio.create_task(waiter('bg2', PRIORITY_BACKGROUND))]
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(waiter('user', PRIORITY_USER)))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(first, *tasks)
    assert order == ['user', 'bg1', 'bg2']


async def test_cancelled_waiters_do_not_leak_slots():
    scheduler = LLMScheduler(1)
    gate = asyncio.Event()

    async def holder():
        async with scheduler.slot():
            await gate.wait()

    async def waiter():
        async with scheduler.slot():
            pass

    first = asyncio.create_task(holder())
    await asyncio.sleep(0)
    second = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    gate.set()
    await first
    assert scheduler.active == 0
    async with scheduler.slot():
        assert scheduler.active == 1


# --- client ------------------------------------------------------------------
async def test_client_strips_think_blocks_and_sends_auth():
    fake = FakeLLM()
    fake.queue('unknown', '<think>內心戲</think>答案')
    seen = {}

    async def handler(request):
        seen['auth'] = request.headers.get('authorization')
        return await fake.handler(request)

    client = LLMClient(httpx.MockTransport(handler))
    result = await client.chat(ENDPOINT, MSG, timeout=5)
    assert result.text == '答案' and seen['auth'] == 'Bearer sekret'


@pytest.mark.parametrize('reply, message', [
    ('HTTP500', 'HTTP 500'),
    (httpx.ConnectError('refused sekret'), '連線失敗'),
])
async def test_client_errors_are_reported_without_leaking_the_key(reply, message):
    fake = FakeLLM()
    fake.queue('unknown', reply)
    client = LLMClient(fake.transport())
    with pytest.raises(LLMError) as error:
        await client.chat(ENDPOINT, MSG, timeout=5)
    assert message in str(error.value) and 'sekret' not in str(error.value)


async def test_client_rejects_truncated_and_empty_output():
    def respond(finish, content):
        async def handler(request):
            return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': finish}]})
        return LLMClient(httpx.MockTransport(handler))

    with pytest.raises(LLMError, match='token'):
        await respond('length', 'partial').chat(ENDPOINT, MSG, timeout=5)
    with pytest.raises(LLMError, match='空白'):
        await respond('stop', '<think>x</think>').chat(ENDPOINT, MSG, timeout=5)


# --- caller ------------------------------------------------------------------
def make_caller(fake, **kw):
    records = []
    caller = LLMCaller(LLMClient(fake.transport()), LLMScheduler(2), timeout=lambda: 5, retries=lambda: 3,
                       logger=records.append, backoff=0, **kw)
    return caller, records


async def test_caller_retries_validation_failures_and_tells_the_model_why():
    fake = FakeLLM()
    fake.queue('unknown', 'bad', 'bad', 'good')
    caller, records = make_caller(fake)
    seen_errors = []

    def build(last_error):
        seen_errors.append(last_error)
        return MSG

    def validate(text):
        if text != 'good':
            raise ValidationError('格式不對')
        return text.upper()

    result = await caller.call(agent='x', endpoint=ENDPOINT, messages=build, validator=validate)
    assert result.value == 'GOOD' and result.attempts == 3 and len(result.call_ids) == 3
    assert seen_errors == [None, '格式不對', '格式不對']
    assert [r['error'] for r in records] == ['格式不對', '格式不對', None]
    assert all(r['messages'] == MSG for r in records)           # the full input is logged every attempt


async def test_caller_gives_up_after_the_configured_retries():
    fake = FakeLLM()
    fake.queue('unknown', *['HTTP500'] * 10)
    caller, records = make_caller(fake)
    with pytest.raises(CallFailed) as error:
        await caller.call(agent='x', endpoint=ENDPOINT, messages=MSG)
    assert len(error.value.call_ids) == 4 and len(records) == 4      # 1 try + 3 retries
    assert 'HTTP 500' in error.value.last_error


async def test_caller_retries_zero_means_a_single_attempt():
    fake = FakeLLM()
    fake.queue('unknown', 'HTTP500', 'good')
    caller, records = make_caller(fake)
    with pytest.raises(CallFailed):
        await caller.call(agent='x', endpoint=ENDPOINT, messages=MSG, retries=0)
    assert len(records) == 1


async def test_cancelled_call_is_logged_and_propagates():
    fake = FakeLLM()
    fake.delay = 5
    caller, records = make_caller(fake)
    task = asyncio.create_task(caller.call(agent='x', endpoint=ENDPOINT, messages=MSG))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert records and records[-1]['error'] == '已取消'


# --- context guard -----------------------------------------------------------
def test_estimate_tokens_is_pessimistic_for_chinese():
    from mini.textutil import estimate_messages_tokens, estimate_tokens
    assert estimate_tokens('') == 0
    assert estimate_tokens('中' * 100) == 100                       # 1 token per CJK character
    assert estimate_tokens('abcde') == 2 and estimate_tokens('a' * 100) == 40
    assert estimate_messages_tokens([{'role': 'user', 'content': '中' * 10}]) == 10 + 8 + 3


async def test_oversized_prompt_fails_before_any_request_is_sent():
    fake = FakeLLM()
    caller, records = make_caller(fake)
    small = Endpoint(context_tokens=1000, max_tokens=500)
    huge = [{'role': 'user', 'content': '中' * 800}]
    with pytest.raises(CallFailed) as error:
        await caller.call(agent='x', endpoint=small, messages=huge)
    assert fake.calls == []                                          # nothing reached the server
    assert '811' in error.value.last_error and 'context 上限' in error.value.last_error
    assert len(error.value.call_ids) == 1 and len(records) == 1
    assert records[0]['messages'] == huge and 'context 上限' in records[0]['error']   # still audited


async def test_prompt_fitting_the_limit_is_sent_and_zero_disables_the_check():
    fake = FakeLLM()
    fake.queue('unknown', 'ok', 'ok')
    caller, _ = make_caller(fake)
    fits = Endpoint(context_tokens=1000, max_tokens=100)
    assert (await caller.call(agent='x', endpoint=fits, messages=[{'role': 'user', 'content': '中' * 800}])).text == 'ok'
    unlimited = Endpoint(context_tokens=0, max_tokens=100)
    await caller.call(agent='x', endpoint=unlimited, messages=[{'role': 'user', 'content': '中' * 50_000}])
    assert len(fake.calls) == 2


async def test_per_call_max_tokens_override_counts_toward_the_limit():
    fake = FakeLLM()
    caller, _ = make_caller(fake)
    endpoint = Endpoint(context_tokens=2000, max_tokens=100)
    with pytest.raises(CallFailed):
        await caller.call(agent='x', endpoint=endpoint, messages=[{'role': 'user', 'content': '中' * 100}],
                          max_tokens=1950)
    assert fake.calls == []


@pytest.mark.parametrize('detail', [
    'request (9000 tokens) exceeds the available context size (8192 tokens)',
    'This model\'s maximum context length is 4096 tokens',
    'n_ctx exceeded',
])
async def test_server_context_overflow_is_reported_once_without_retries(detail):
    calls = []

    async def handler(request):
        calls.append(1)
        return httpx.Response(400, json={'error': {'message': detail}})

    caller = LLMCaller(LLMClient(httpx.MockTransport(handler)), LLMScheduler(2), timeout=lambda: 5,
                       retries=lambda: 3, logger=lambda r: None, backoff=0)
    with pytest.raises(CallFailed) as error:
        await caller.call(agent='x', endpoint=Endpoint(context_tokens=0), messages=MSG)
    assert len(calls) == 1                                           # retrying the same prompt cannot help
    assert '模型設定' in error.value.last_error


async def test_other_server_errors_are_still_retried():
    fake = FakeLLM()
    fake.queue('unknown', 'HTTP500', 'good')
    caller, records = make_caller(fake)
    result = await caller.call(agent='x', endpoint=ENDPOINT, messages=MSG)
    assert result.text == 'good' and len(records) == 2
