"""Source tags for the daily record (NOTE).

Allowed tags:  [語音#N] [語音#N-M] [語音#N,M]   [醫師手動]   [歷史]

As in the original TCM-Meridian, the program does only two things with them: it appends `[醫師手動]` to the
lines a physician edits by hand (`tag_human_edits`), and it hides the tags in the browse view and when copying
(`strip_citations`). Whether the tags the writer agent produces are right is NOT checked by code; that is the
job of the prompts and the hallucination reviewer (an LLM), exactly as in the original.
"""
from __future__ import annotations

import difflib
import re

MANUAL_TAG = '[醫師手動]'
_TAG_BODY = r'(?:語音#\d+(?:[-,]\d+)*|醫師手動|歷史)'


def strip_citations(text: str) -> str:
    """Hide source tags (browse view and copy)."""
    return re.sub(rf'\s*\[{_TAG_BODY}\]', '', text)


def _append_manual(line: str) -> str:
    # As in the original project: a tag already at the END of the line is not added again. So consecutive manual
    # edits leave ONE tag; a second one appears whenever there is anything after the first tag when the line is
    # edited again -- an AI-appended source (`…[醫師手動][語音#8]`) or text the physician typed after it. A repeated
    # tag therefore means "edited while the tag was not last", NOT a count of edits: the 病歷修改 diff 過程 has
    # the full history.
    return line if line.rstrip().endswith(MANUAL_TAG) else line + MANUAL_TAG


def tag_human_edits(old_text: str, new_text: str) -> str:
    """Append [醫師手動] to every added/changed non-empty line unless the line already ends with it."""
    old_lines = (old_text or '').split('\n')
    new_lines = (new_text or '').split('\n')
    result: list[str] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
        for line in new_lines[j1:j2]:
            result.append(_append_manual(line) if op != 'equal' and line.strip() else line)
    return '\n'.join(result)
