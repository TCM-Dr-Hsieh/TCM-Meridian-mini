"""Source tags for the daily record (NOTE).

Allowed tags:  [語音#N] [語音#N-M] [語音#N, #M]   [醫師手動]   [歷史]

As in the original TCM-Meridian, the program appends `[醫師手動]` to the lines a physician edits by hand
(`tag_human_edits`), and it hides the tags in the browse view and when copying (`strip_citations`). Whether the tags the
writer agent produces are right is NOT checked by code; that is the job of the prompts and the hallucination reviewer
(an LLM), exactly as in the original.

The edit box does not show the tags (`editable_text`), so the physician cannot damage them. On save,
`merge_human_edit` puts them back by diffing the old text against the edited one, and takes a `[語音#N]` / `[歷史]` tag
away when the words it covered are gone -- the one place where the program removes a source tag (see its docstring).
`[醫師手動]` is never taken away while its line exists (it goes with the line when the line is deleted). Tag syntax in
the edit box is refused (`NoteEditError`): the note format cannot tell a
typed `[語音#9]` from a real one, so it must not get in.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

MANUAL_TAG = '[醫師手動]'


class NoteEditError(RuntimeError):
    """A hand edit of the note that is refused; the message is for the physician."""


# A voice tag is one segment number followed by any further numbers: a range (`-4`) or a list (`, #4`). The prompts teach
# `[語音#N-M]` and `[語音#N, #M]` (the list used to be documented as `[語音#N,M]`, but models write `, #M`, which that
# spelling did not hide). Hiding is deliberately lenient: it also takes `,#4`, `,4` (what earlier visits' notes carry), a
# full-width comma or 頓號, and an en or em dash. Only the hiding is lenient; whether a tag is right is still not checked.
_SEGMENTS = r'\d+(?:\s*[-–—,，、]\s*#?\d+)*'
_TAG_BODY = rf'(?:語音#{_SEGMENTS}|醫師手動|歷史)'


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


# ---------------------------------------------------------------------------------------------------------------------
# The edit box shows the note WITHOUT its tags; on save the tags are put back (and some taken away) by the program.
# ---------------------------------------------------------------------------------------------------------------------
# A run of fewer than this many letters/digits that is left over after an edit is treated as chance (the physician typed
# "大便" again while rewriting "大便1天2次，偏軟"); it does not keep a tag alive. A span with fewer letters than this
# needs all of them (a two-letter span the physician only extended must keep its tag).
MIN_SURVIVING_RUN = 3
_BARE_TAG_RE = re.compile(rf'\[{_TAG_BODY}\]')
_TAG_RE = re.compile(rf'[ \t\u3000]*\[{_TAG_BODY}\]')       # a tag together with the blanks in front of it


def typed_tags(text: str) -> list[str]:
    """Source tags in `text` (what the physician typed or pasted into the edit box, which shows none), in order, once each."""
    return list(dict.fromkeys(_BARE_TAG_RE.findall(text or '')))


@dataclass(frozen=True)
class RemovedTag:
    tag: str            # e.g. '[語音#7]'
    text: str           # the words it covered (as they were before the edit), for the audit and the preview


@dataclass(frozen=True)
class EditMerge:
    text: str                           # the new note, tags included
    removed: tuple[RemovedTag, ...] = ()    # source tags that are gone ([醫師手動] is not listed: it is added back anyway)


def tokenize(text: str) -> list[tuple[str, bool]]:
    """`text` as [(piece, is_tag)]: the tags (as written) and the text between them."""
    pieces, pos = [], 0
    for match in _BARE_TAG_RE.finditer(text):
        if match.start() > pos:
            pieces.append((text[pos:match.start()], False))
        pieces.append((match.group(), True))
        pos = match.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def _split_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """(the text without its tags, [(offset in that text, the tag with the blanks in front of it)])."""
    plain: list[str] = []
    tags: list[tuple[int, str]] = []
    pos = length = 0
    for match in _TAG_RE.finditer(text):
        chunk = text[pos:match.start()]
        plain.append(chunk)
        length += len(chunk)
        tags.append((length, match.group()))
        pos = match.end()
    plain.append(text[pos:])
    return ''.join(plain), tags


def editable_text(note: str) -> str:
    """What the edit box shows: the note without its source tags."""
    return _split_tags(note or '')[0]


def _content_mask(plain: str) -> list[bool]:
    """Per character: does it carry the source's words? Letters (CJK included) and digits do; punctuation and blanks do not.
    Templates differ from physician to physician, so no part of a line is assumed to be a field name: one that is left
    standing counts as words, and a tag may then stay when its sentence was rewritten (kept too long, never lost)."""
    return [ch.isalnum() for ch in plain]


def _groups(plain: str, tags: list[tuple[int, str]]) -> list[tuple[int, int, list[str]]]:
    """[(start, offset, tags)]: tags written one after another (`…[語音#1][語音#2]`) are one group; it covers the text
    from the previous group to its own position."""
    grouped: list[tuple[int, list[str]]] = []
    for offset, token in tags:
        if grouped and grouped[-1][0] == offset:
            grouped[-1][1].append(token)
        else:
            grouped.append((offset, [token]))
    out, start = [], 0
    for offset, tokens in grouped:
        out.append((start, offset, tokens))
        start = offset
    return out


def _survives(ops, mask: list[bool], start: int, end: int, min_run: int = MIN_SURVIVING_RUN) -> bool:
    """Is the text of [start:end) of the old text still there after the edit (`ops`: old text -> new)? A span that had no
    words to begin with (only punctuation, or nothing: two tags side by side) has nothing to lose."""
    total = sum(mask[start:end])
    if total == 0:
        return True
    need = min(min_run, total)
    for op, i1, i2, _j1, _j2 in ops:
        if op != 'equal':
            continue
        lo, hi = max(i1, start), min(i2, end)
        if lo < hi and sum(mask[lo:hi]) >= need:
            return True
    return False


def _new_offset(ops, offset: int, new_length: int) -> int:
    """Where a position of the old text is in the new one. A position at the end of unchanged text stays before text
    typed right after it, so a tag keeps covering only what it covered."""
    for op, i1, i2, j1, j2 in ops:
        if not i1 <= offset <= i2:
            continue
        if op == 'equal':
            return j1 + offset - i1
        if offset == i1:
            return j1
        if offset < i2:                         # inside a rewritten stretch: after what replaced it
            return j2 if op == 'replace' else j1
    return new_length


def _snippet(plain: str, start: int, end: int) -> str:
    return re.sub(r'\s+', ' ', plain[start:end]).strip()


def _merge_block(old_block: str, new_block: str, min_run: int = MIN_SURVIVING_RUN) -> tuple[str, list[RemovedTag]]:
    old_plain, tags = _split_tags(old_block)
    if not tags:
        return new_block, []
    ops = difflib.SequenceMatcher(None, old_plain, new_block, autojunk=False).get_opcodes()
    mask = _content_mask(old_plain)
    kept: list[tuple[int, str]] = []
    removed: list[RemovedTag] = []
    floor = 0
    for start, offset, tokens in _groups(old_plain, tags):
        alive = _survives(ops, mask, start, offset, min_run)
        # [醫師手動] is never taken away (the prompts require every one of them to stay, and the audit keeps the history);
        # only the [語音#N] / [歷史] tags go with their words.
        stay = tokens if alive else [token for token in tokens if token.strip() == MANUAL_TAG]
        if stay:
            floor = max(floor, _new_offset(ops, offset, len(new_block)))
            kept.extend((floor, token) for token in stay)
        if not alive:
            text = _snippet(old_plain, start, offset)
            removed.extend(RemovedTag(token.strip(), text) for token in tokens if token.strip() != MANUAL_TAG)
    pieces, prev = [], 0
    for pos, token in kept:
        pieces += [new_block[prev:pos], token]
        prev = pos
    pieces.append(new_block[prev:])
    return ''.join(pieces), removed


def merge_human_edit(old_note: str, edited: str, min_run: int = MIN_SURVIVING_RUN) -> EditMerge:
    """The note after the physician edited `editable_text(old_note)` into `edited`.

    Unchanged lines are kept exactly as they were (tags and all). In the lines that changed:
    * each tag stays where the words it covers are; a tag covers the words from the previous tag up to itself, and tags
      written one after another cover the same words;
    * a `[語音#N]` / `[歷史]` tag is REMOVED when none of the words it covered survive (letters and digits count, punctuation
      does not; a left over run shorter than `min_run` is chance; Settings.note_edit.min_surviving_run, MIN_SURVIVING_RUN
      by default), whatever the physician typed instead: the new words are the physician's, and the old tag no longer
      describes them. Tags with no words in front of them to begin with (side by side) are not touched by this rule;
      `[醫師手動]` tags are never removed while the line exists, however many there are;
    * the line gets `[醫師手動]` at its end, as `tag_human_edits` does.
    Tags of lines deleted outright go with the lines. The edit box shows no tags, so tag syntax in `edited` was typed or
    pasted (the reference window can be copied from): `NoteEditError` -- the note format cannot tell it from a real tag.
    """
    typed = typed_tags(edited)
    if typed:
        raise NoteEditError(f'修改框裡有來源標籤（{"、".join(typed)}）。來源標籤只能由系統加上，不能手打或貼上，'
                            '請先把它刪除（或按「放棄」）再儲存。')
    old_lines = (old_note or '').split('\n')
    new_lines = (edited or '').replace('\r\n', '\n').replace('\r', '\n').split('\n')
    clean_old = [_split_tags(line)[0] for line in old_lines]
    out: list[str] = []
    removed: list[RemovedTag] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, clean_old, new_lines, autojunk=False).get_opcodes():
        if op == 'equal':
            out.extend(old_lines[i1:i2])
            continue
        if op == 'delete':
            plain, tags = _split_tags('\n'.join(old_lines[i1:i2]))
            for start, offset, tokens in _groups(plain, tags):
                removed.extend(RemovedTag(token.strip(), _snippet(plain, start, offset)) for token in tokens)
            continue
        if op == 'replace':
            merged, gone = _merge_block('\n'.join(old_lines[i1:i2]), '\n'.join(new_lines[j1:j2]), min_run)
            removed.extend(gone)
            lines = merged.split('\n')
        else:                                   # insert
            lines = new_lines[j1:j2]
        out.extend(_append_manual(line) if line.strip() else line for line in lines)
    return EditMerge('\n'.join(out), tuple(r for r in removed if r.tag != MANUAL_TAG))
