"""Settings: dataclasses, defaults, validation, and atomic persistence."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

from .fileio import atomic_write_json

DEFAULT_HOST = '0.0.0.0'      # all interfaces: reachable from the LAN and through a tunnel (e.g. cloudflared)
DEFAULT_PORT = 5050


def server_binding(env=None) -> tuple[str, int]:
    """(host, port) the web UI listens on: MINI_HOST (default 0.0.0.0) and MINI_PORT (default 5050).

    Set MINI_HOST=127.0.0.1 to accept this computer only. The app has no login, so anyone who can reach the
    address sees the (single, global) visit and can operate it.
    """
    env = os.environ if env is None else env
    host = (env.get('MINI_HOST') or '').strip() or DEFAULT_HOST
    raw = (env.get('MINI_PORT') or '').strip() or str(DEFAULT_PORT)
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(f'MINI_PORT 必須是整數，目前是「{raw}」。') from None
    if not 1 <= port <= 65535:
        raise ValueError(f'MINI_PORT 必須介於 1 到 65535，目前是 {port}。')
    return host, port


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config.json'
PROMPTS_DIR = ROOT / 'prompts'
TEMPLATES_DIR = ROOT / 'templates'

DEFAULT_URL = 'http://100.85.255.46:8080/v1'
DEFAULT_KEY = 'no-key'
DEFAULT_MODEL = 'qwen3.8-27b'
MAX_VOCABULARY_CHARS = 25_000
DEFAULT_CONTEXT_TOKENS = 125_000
DEFAULT_ASR_DIR = r'models\Qwen3-ASR-1.7B'
DEFAULT_ALIGNER_DIR = r'models\Qwen3-ForcedAligner-0.6B'
DEFAULT_SPEAKER_MODEL = r'models\speaker\3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'

AGENT_KEYS = ('transcript_corrector', 'record_writer', 'hallucination_corrector',
              'ai_advice', 'professor_a', 'professor_b', 'professor_c', 'deidentifier')
AGENT_LABELS = {
    'transcript_corrector': '逐字稿校稿',
    'record_writer': '病歷書寫',
    'hallucination_corrector': '幻覺修正（審查）',
    'ai_advice': '問診建議',
    'professor_a': '教授甲',
    'professor_b': '教授乙',
    'professor_c': '教授丙（仲裁）',
    'deidentifier': '去識別化',
}
# (temperature, max_tokens); max_tokens 0 means "chosen per request".
AGENT_DEFAULTS = {
    'transcript_corrector': (0.2, 0),
    'record_writer': (0.7, 8000),
    'hallucination_corrector': (0.7, 4000),
    'ai_advice': (0.7, 4000),
    'professor_a': (0.7, 8000),
    'professor_b': (0.7, 8000),
    'professor_c': (0.3, 8000),
    'deidentifier': (0.2, 0),        # the reply is about as long as the input: sized per request
}

DEFAULT_STYLE_A = ('你以嚴謹、保守、重視風險的風格撰寫：優先排除危險徵象與用藥禁忌；對證據不足的推論明確標示不確定，'
                   '不輕易下肯定結論；處方與衛教寧可保守，並明列需留意的禁忌與追蹤條件。語氣直接、不迂迴，但須客觀，不得攻擊個人。')
DEFAULT_STYLE_B = ('你以溫和、循序漸進的風格撰寫：重視患者的耐受性、接受度與生活可行性；處方與衛教從簡單、安全、易執行的方案開始，'
                   '並說明如何逐步調整。語氣平和、鼓勵，但不得因溫和而忽略危險徵象與必要轉診。')


def resolve_path(value) -> Path:
    """Expand ~ and resolve relative paths against the project root (never the process CWD)."""
    path = Path(str(value).strip().strip('"')).expanduser()
    return path if path.is_absolute() else ROOT / path


def _number(value, name, low=None, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 須為數值。')
    if integer and float(value) != int(value):
        raise ValueError(f'{name} 須為整數。')
    if low is not None and value < low:
        raise ValueError(f'{name} 不可小於 {low:g}。')
    if high is not None and value > high:
        raise ValueError(f'{name} 不可大於 {high:g}。')
    return int(value) if integer else float(value)


@dataclass
class Endpoint:
    api_url: str = DEFAULT_URL
    api_key: str = DEFAULT_KEY
    model_name: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 4000
    # The model's context window in tokens (prompt + reply). Prompts that would not fit fail early with a clear
    # message instead of failing at the server. 0 = do not check. Servers rarely report this, so set the real value.
    context_tokens: int = DEFAULT_CONTEXT_TOKENS

    def validate(self, label: str = 'LLM'):
        self.api_url = self.api_url.strip().rstrip('/，, ')
        self.model_name = self.model_name.strip()
        url = urlsplit(self.api_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment:
            raise ValueError(f'{label}：API URL 必須是完整的 http(s) 網址，不含查詢或片段。')
        if not self.model_name:
            raise ValueError(f'{label}：模型名稱不可留空。')
        self.temperature = _number(self.temperature, f'{label} temperature', 0, 2)
        self.max_tokens = _number(self.max_tokens, f'{label} max_tokens', 0, 200_000, integer=True)
        self.context_tokens = _number(self.context_tokens, f'{label} context 上限', 0, 4_000_000, integer=True)


@dataclass
class LLMSettings:
    max_concurrency: int = 2
    timeout_seconds: float = 180.0
    retries: int = 3

    def validate(self):
        self.max_concurrency = _number(self.max_concurrency, 'LLM 並行上限', 1, 32, integer=True)
        self.timeout_seconds = _number(self.timeout_seconds, 'LLM 逾時', 5, 1800)
        self.retries = _number(self.retries, 'LLM 重試次數', 0, 10, integer=True)


@dataclass
class ASRSettings:
    asr_model_dir: str = DEFAULT_ASR_DIR            # relative paths are resolved against the project folder
    aligner_model_dir: str = DEFAULT_ALIGNER_DIR
    device: str = 'auto'
    window_seconds: float = 6.0
    overlap_seconds: float = 3.0
    context_chars: int = 1000
    vocabulary: str = ''                  # hint for the speech recognizer only
    correction_vocabulary: str = ''       # spelling reference for the transcript-correction LLM only
    microphone_id: str = ''
    python_path: str = ''

    def validate(self):
        self.asr_model_dir = self.asr_model_dir.strip().strip('"')
        self.aligner_model_dir = self.aligner_model_dir.strip().strip('"')
        if not self.asr_model_dir:
            raise ValueError('ASR 模型資料夾不可留空。')
        if not self.aligner_model_dir:
            raise ValueError('ForcedAligner 模型資料夾不可留空。')
        if self.device not in ('auto', 'cuda', 'cpu'):
            raise ValueError('ASR 運算裝置須為 auto、cuda 或 cpu。')
        self.window_seconds = _number(self.window_seconds, '音訊視窗長度', 5, 30)
        self.overlap_seconds = _number(self.overlap_seconds, '重疊秒數', 0)
        if self.overlap_seconds > self.window_seconds - 3:
            raise ValueError('重疊須小於等於「視窗長度 − 3」，至少保留 3 秒新音訊。')
        self.context_chars = _number(self.context_chars, '校稿前文長度', 100, 20000, integer=True)
        if len(self.vocabulary) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'ASR 專有詞最多 {MAX_VOCABULARY_CHARS} 字元。')
        if len(self.correction_vocabulary) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'校稿 LLM 專有詞最多 {MAX_VOCABULARY_CHARS} 字元。')
        self.python_path = self.python_path.strip().strip('"')


@dataclass
class SpeakerSettings:
    """Speaker marking (SPEC 4.3): label the transcript 醫師 / 患者或家屬 / 不明 from the voices."""
    enabled: bool = False
    model_path: str = DEFAULT_SPEAKER_MODEL          # relative paths are resolved against the project folder
    use_in_jobs: bool = True                         # the writer and reviewer prompts get the speaker marks
    text_fill: bool = True                           # an LLM decides some of the sentences the voice could not
    text_fill2: bool = False                         # ... and a second, blind pass over what is still unknown: two equal answers decide
    unknown_percentile: float = 15.0                 # higher = more "不明", fewer mistakes

    def validate(self):
        self.enabled = bool(self.enabled)
        self.use_in_jobs = bool(self.use_in_jobs)
        self.text_fill = bool(self.text_fill)
        self.text_fill2 = bool(self.text_fill2) and self.text_fill         # the second pass cannot work without the first
        self.model_path = str(self.model_path).strip().strip('"')
        if not self.model_path:
            raise ValueError('說話者聲紋模型路徑不可留空。')
        self.unknown_percentile = _number(self.unknown_percentile, '「不明」百分位', 0, 50)


@dataclass
class ReviewSettings:
    pass_required_n: int = 2      # 0 = no review (control-group mode)
    max_review_rounds: int = 6

    def validate(self):
        self.pass_required_n = _number(self.pass_required_n, '審查通過次數 n', 0, 50, integer=True)
        self.max_review_rounds = _number(self.max_review_rounds, '最大審查輪數', 1, 50, integer=True)
        if self.pass_required_n > self.max_review_rounds:
            raise ValueError('審查通過次數 n 不可大於最大審查輪數。')


@dataclass
class ProfessorStyle:
    name: str = ''
    role_style: str = ''


def default_agents() -> dict[str, Endpoint]:
    return {key: Endpoint(temperature=AGENT_DEFAULTS[key][0], max_tokens=AGENT_DEFAULTS[key][1])
            for key in AGENT_KEYS}


def default_professors() -> dict[str, ProfessorStyle]:
    return {'a': ProfessorStyle('教授甲', DEFAULT_STYLE_A),
            'b': ProfessorStyle('教授乙', DEFAULT_STYLE_B),
            'c': ProfessorStyle('教授丙', '')}


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    speaker: SpeakerSettings = field(default_factory=SpeakerSettings)
    review: ReviewSettings = field(default_factory=ReviewSettings)
    agents: dict[str, Endpoint] = field(default_factory=default_agents)
    professors: dict[str, ProfessorStyle] = field(default_factory=default_professors)
    visits_dir: str = 'visits'

    def validate(self):
        self.llm.validate()
        self.asr.validate()
        self.speaker.validate()
        self.review.validate()
        for key in AGENT_KEYS:
            self.agents[key].validate(AGENT_LABELS[key])
        for key in ('a', 'b', 'c'):
            style = self.professors[key]
            style.name = style.name.strip() or default_professors()[key].name
        if not self.visits_dir.strip():
            raise ValueError('看診資料夾不可留空。')

    def visits_path(self) -> Path:
        return resolve_path(self.visits_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    def public_dict(self) -> dict:
        """Settings snapshot for audit files: never includes API keys."""
        data = self.to_dict()
        for endpoint in data['agents'].values():
            endpoint.pop('api_key', None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Settings':
        if not isinstance(data, dict):
            raise ValueError('設定檔最外層必須是 JSON 物件。')

        def build(klass, raw):
            names = {item.name for item in fields(klass)}
            return klass(**{k: v for k, v in (raw if isinstance(raw, dict) else {}).items() if k in names})

        settings = cls()
        settings.llm = build(LLMSettings, data.get('llm'))
        raw_asr = data.get('asr') if isinstance(data.get('asr'), dict) else {}
        settings.asr = build(ASRSettings, raw_asr)
        if 'correction_vocabulary' not in raw_asr:
            # A config from before the vocabulary was split into two: the corrector used to receive the one shared list,
            # so start it from that instead of silently taking the list away. Once saved, the two are independent.
            settings.asr.correction_vocabulary = settings.asr.vocabulary
        settings.speaker = build(SpeakerSettings, data.get('speaker'))
        settings.review = build(ReviewSettings, data.get('review'))
        agents = default_agents()
        for key, raw in (data.get('agents') or {}).items():
            if key in agents and isinstance(raw, dict):
                merged = asdict(agents[key])
                merged.update({k: v for k, v in raw.items() if k in merged})
                agents[key] = Endpoint(**merged)
        raw_agents = data.get('agents') if isinstance(data.get('agents'), dict) else {}
        if 'deidentifier' not in raw_agents and isinstance(raw_agents.get('record_writer'), dict):
            # A config from before the 去識別化 interface existed. It receives identifiable patient data, so it must not
            # silently fall back to the built-in default address: start it from the interface the physician already
            # chose for writing the record (address, key, model, context window). Once saved, the two are independent.
            writer = agents['record_writer']
            agents['deidentifier'] = Endpoint(
                api_url=writer.api_url, api_key=writer.api_key, model_name=writer.model_name,
                temperature=AGENT_DEFAULTS['deidentifier'][0], max_tokens=AGENT_DEFAULTS['deidentifier'][1],
                context_tokens=writer.context_tokens)
        settings.agents = agents
        professors = default_professors()
        for key, raw in (data.get('professors') or {}).items():
            if key in professors and isinstance(raw, dict):
                professors[key] = ProfessorStyle(str(raw.get('name', professors[key].name)),
                                                 str(raw.get('role_style', professors[key].role_style)))
        settings.professors = professors
        if isinstance(data.get('visits_dir'), str):
            settings.visits_dir = data['visits_dir']
        settings.validate()
        return settings

    def save(self, path: Path = CONFIG_PATH):
        self.validate()
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> 'Settings':
        """Load settings; the first run writes defaults to disk."""
        if not path.exists():
            settings = cls()
            settings.save(path)
            return settings
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
