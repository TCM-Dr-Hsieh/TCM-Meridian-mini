"""Rule-based scan of a de-identified text for identifiers the model left behind.

De-identification by an LLM is not guaranteed to be complete, so every result is scanned for what a rule can see: the
visit date written out, ID / phone / e-mail / address shapes, dates that should have become relative intervals, a
surname with an honorific, and values copied verbatim from the input's own 姓名 field or long digit strings. It never
blocks a result and never rewrites it; it only returns warnings for the physician. A warning names the kind and the
count, not the matched text, so the identifier is not repeated in a second place.
"""
from __future__ import annotations

import re
from datetime import date

_CJK = '一-鿿'

# kind -> pattern. Each is a *shape* of an identifier; a clean result contains none of them.
SHAPES: tuple[tuple[str, re.Pattern], ...] = (
    ('身分證或居留證字號', re.compile(r'(?<![A-Za-z0-9])[A-Za-z][12ABCDabcd89]\d{8}(?!\d)')),
    ('手機號碼', re.compile(r'(?<!\d)(?:\+?886[-\s]?|0)9\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)')),
    ('市話號碼', re.compile(r'(?<!\d)\(?0[2-8]\d?\)?[-\s]?\d{3,4}[-\s]?\d{3,4}(?!\d)')),
    ('Email', re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')),
    ('網址', re.compile(r'https?://\S+|www\.\S+', re.I)),
    ('病歷號、健保卡號或證號', re.compile(r'(?:病歷號|病歷編號|病例號|健保卡號|卡號|掛號號|報告單號|chart\s*no|MRN)'
                                          r'\s*[:：#]?\s*[A-Za-z]*\d{4,}', re.I)),
    ('門牌地址', re.compile(rf'[{_CJK}]{{1,8}}(?:路|街|大道)(?:[一二三四五六七八九十\d]+段)?(?:\d+巷)?(?:\d+弄)?\d+號')),
)

# Anything that names a calendar date more precisely than a year. Month-and-year counts too: the rule is intervals.
# Not scanned: 月/日 with a slash (10/20 looks like a dose, 1/2 包) and dates separated by spaces.
_MONTH_DAY = r'(?:0?[1-9]|1[0-2])'
_DAY = r'(?:0?[1-9]|[12]\d|3[01])'
_MM_DD = r'(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])'
DATES = re.compile('|'.join((
    rf'(?<!\d)(?:19|20)\d{{2}}{_MM_DD}(?!\d)',                                       # 20260829
    rf'(?<!\d)1\d{{2}}{_MM_DD}(?!\d)',                                               # 1150829 (民國)
    rf'(?<!\d)(?:19|20)\d{{2}}\s*[-/.]\s*{_MONTH_DAY}\s*[-/.]\s*{_DAY}(?!\d)',      # 2026-08-29
    rf'(?<![\d.])\d{{2,3}}\s*[/.]\s*{_MONTH_DAY}\s*[/.]\s*{_DAY}(?![\d.])',        # 115/08/29 (民國)
    rf'\d{{2,4}}\s*年\s*{_MONTH_DAY}\s*月(?:\s*{_DAY}\s*日)?',                      # 115年8月29日, 2019年3月
    rf'(?<!\d){_MONTH_DAY}\s*月\s*{_DAY}\s*日',                                    # 8月29日
)))

# 「王先生」 names a surname; 「患者太太」 「他先生」 「某某小姐」 only name a relation or are already anonymous. A real
# surname is never one of these characters, so a prefix ending in one is not flagged.
HONORIFIC = re.compile(rf'([{_CJK}]{{1,2}})(先生|小姐|太太|女士)')
GENERIC_LAST = frozenset('者人患他她其我你您的某該此貴本男女')

NAME_FIELD = re.compile(rf'(?:病患姓名|患者姓名|姓名|名字)\s*[:：]\s*([{_CJK}]{{2,5}})')
LONG_NUMBER = re.compile(r'\d+(?:-\d+)*')
MIN_LONG_DIGITS = 7
MIN_NOTE_SHARE = 0.5          # a note much shorter than its source was probably summarised
MIN_NOTE_CHARS = 200


def _spans(pattern: re.Pattern, text: str) -> list[tuple[int, int]]:
    return [match.span() for match in pattern.finditer(text)]


def visit_date_pattern(day: date) -> re.Pattern:
    """Today's date in any of the usual written forms, Gregorian or 民國, padded or not, with or without separators."""
    year = rf'(?:{day.year}|{day.year - 1911})'
    month, mday = f'0?{day.month}', f'0?{day.day}'
    separated = rf'(?<!\d){year}\s*[-/.年]\s*{month}\s*[-/.月]\s*{mday}(?!\d)'
    compact = rf'(?<!\d){year}{day.month:02d}{day.day:02d}(?!\d)'
    month_day = rf'(?<!\d){month}\s*月\s*{mday}\s*日'
    return re.compile('|'.join((separated, compact, month_day)))


def _digit_runs(text: str) -> set[str]:
    runs = set()
    for match in LONG_NUMBER.finditer(text):
        digits = match.group().replace('-', '')
        if len(digits) >= MIN_LONG_DIGITS:
            runs.add(digits)
    return runs


def residual_warnings(*, visit_date: date, source_patient: str, source_note: str,
                      date_text: str, patient_text: str, note_text: str) -> list[str]:
    """Warnings (Traditional Chinese, one line each) for what the de-identified text still shows; [] when it looks clean."""
    output = '\n'.join((date_text, patient_text, note_text))
    source = source_patient + '\n' + source_note
    warnings: list[str] = []

    today = visit_date_pattern(visit_date)
    today_spans = _spans(today, output)
    if today_spans:
        warnings.append(f'今日看診日期的原文仍出現在結果中（{len(today_spans)} 處）：應只寫「D日」。')

    for kind, pattern in SHAPES:
        count = len(pattern.findall(output))
        if count:
            warnings.append(f'疑似仍含{kind}（{count} 處）。')

    dates = [span for span in _spans(DATES, output)
             if not any(span[0] < end and start < span[1] for start, end in today_spans)]
    if dates:
        warnings.append(f'疑似仍含具體日期（{len(dates)} 處）：其他日期應改成相對今日的粗略時距（如「約 5 週前」）。')

    honorifics = [m for m in HONORIFIC.finditer(output) if m.group(1)[-1] not in GENERIC_LAST]
    if honorifics:
        warnings.append(f'疑似仍含「姓氏＋稱謂」（{len(honorifics)} 處，如「王先生」「李小姐」）。')

    names = {m.group(1) for m in NAME_FIELD.finditer(source)}
    if any(name in output for name in names):
        warnings.append('結果仍含輸入資料中「姓名」欄位的內容。')

    carried = _digit_runs(source) & _digit_runs(output)
    if carried:
        warnings.append(f'輸入資料中的長數字串原樣出現在結果中（{len(carried)} 組，可能是證號、病歷號或電話）。')

    if len(source_note.strip()) >= MIN_NOTE_CHARS and len(note_text.strip()) < len(source_note.strip()) * MIN_NOTE_SHARE:
        warnings.append('今日病歷的長度不到原文的一半：可能被摘要或遺漏了臨床內容，請對照原文確認。')
    return warnings
