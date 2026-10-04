import asyncio

import pytest

from mini.llm import LLMClient, LLMScheduler
from mini.visit import VisitSession
from mini.visit_store import VisitStore
from tests.helpers import FakeASR, FakeLLM, make_settings, silent_source


@pytest.fixture
def fake_llm():
    return FakeLLM()


class Harness:
    """A started VisitSession wired to fakes, with the transcript already processed."""

    def __init__(self, session, fake, settings):
        self.session, self.fake, self.settings = session, fake, settings

    async def run_job(self, kind: str):
        job = self.session.start_job(kind)
        await asyncio.wait_for(job.task, 20)
        return job


@pytest.fixture
def make_session(tmp_path, fake_llm):
    created: list[VisitSession] = []

    async def factory(*, seconds=7, asr_text='我頭痛三天了', patient='45歲男性，高血壓病史', limit=2, **overrides):
        settings = make_settings(tmp_path)
        settings.llm.max_concurrency = limit
        for key, value in overrides.items():
            obj, attr = key.split('__')
            setattr(getattr(settings, obj), attr, value)
        settings.validate()
        store = VisitStore.allocate(settings.visits_path())
        session = VisitSession(
            store=store, settings=settings, patient_text=patient, record_template='甲- 現病史：\n乙- 過去病史：',
            analysis_template='一- 西醫診斷：\n二- 中醫診斷：', client=LLMClient(fake_llm.transport()),
            scheduler=LLMScheduler(limit), asr=FakeASR(default=asr_text),
            source_factory=lambda path: silent_source(settings, seconds, recording_path=path),
            llm_backoff=0)
        await session.start()
        await session.pipeline.wait_finished(timeout=20)
        created.append(session)
        return Harness(session, fake_llm, settings)

    yield factory
