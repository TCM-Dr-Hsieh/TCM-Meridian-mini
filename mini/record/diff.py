"""Diff helpers: HTML for the UI, unified text for prompts, and the readable diff.md."""
from __future__ import annotations

import difflib
import html
import re

from ..textutil import format_clock


def unified_diff(old_text: str, new_text: str, old_label: str = '舊版', new_label: str = '新版',
                 context: int = 3) -> str:
    return '\n'.join(difflib.unified_diff((old_text or '').splitlines(), (new_text or '').splitlines(),
                                          fromfile=old_label, tofile=new_label, lineterm='', n=context))


def _version_label(number: int, snapshot: dict) -> str:
    source = snapshot.get('source') or 'unknown'
    stamp = snapshot.get('timestamp') or ''
    return f'版本 {number}（{source}, {stamp}）' if stamp else f'版本 {number}（{source}）'


def build_diff_context(snapshots: list[dict] | None, current_index: int | None = None) -> str:
    """The 【病歷修改 diff 過程】 prompt body (ported from TCM-Meridian record_diff_context).

    One unified diff per consecutive pair of versions, from the initial version up to the version the physician
    is looking at (`current_index`; redo-able versions beyond it are not part of the history). Each version is
    labelled with its source ('醫師手動', '病歷書寫 #k', 'init') and time, so the agent can tell what a human
    typed and how the existing lines came to be.
    """
    valid = [snap for snap in (snapshots or []) if isinstance(snap, dict)]
    if current_index is None:
        current_index = len(valid) - 1
    if current_index < 0 or not valid:
        return '（無病歷版本歷史）'
    active = valid[:max(0, min(current_index, len(valid) - 1)) + 1]
    if len(active) < 2:
        return '（目前只有一個版本，尚無版本間 diff）'
    parts = []
    for index in range(1, len(active)):
        before, after = active[index - 1], active[index]
        diff = unified_diff(before.get('note', ''), after.get('note', ''),
                            f'{_version_label(index, before)} NOTE', f'{_version_label(index + 1, after)} NOTE')
        if diff:
            parts.append(f'### 版本 {index} -> 版本 {index + 1}：NOTE\n```diff\n{diff}\n```')
    return '\n\n'.join(parts) or '（目前版本歷史中沒有內容差異）'


def simple_md_render(text: str) -> str:
    """Small markdown subset rendered as safe HTML (input is escaped first)."""
    escaped = html.escape(text)
    escaped = re.sub(r'^### (.+)$', r'<div class="md-h3">\1</div>', escaped, flags=re.M)
    escaped = re.sub(r'^## (.+)$', r'<div class="md-h2">\1</div>', escaped, flags=re.M)
    escaped = re.sub(r'^# (.+)$', r'<div class="md-h1">\1</div>', escaped, flags=re.M)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    return escaped.replace('\n', '<br>')


def diff_html(old_text: str, new_text: str, title: str = '') -> str:
    """Unified diff as colored HTML (ported from TCM-Meridian rendering.generate_diff_html)."""
    parts = []
    if title:
        parts.append(f'<div class="diff-title">{html.escape(title)}</div>')
    parts.append('<div class="diff-box">')
    has_diff = False
    for line in difflib.unified_diff((old_text or '').splitlines(), (new_text or '').splitlines(),
                                     lineterm='', n=3):
        escaped = html.escape(line)
        if line.startswith('+++') or line.startswith('---'):
            continue
        if line.startswith('@@'):
            parts.append(f'<span class="diff-hunk">{escaped}</span>\n')
            has_diff = True
        elif line.startswith('+'):
            parts.append(f'<span class="diff-add">{escaped}</span>\n')
            has_diff = True
        elif line.startswith('-'):
            parts.append(f'<span class="diff-del">{escaped}</span>\n')
            has_diff = True
        else:
            parts.append(f'{escaped}\n')
    if not has_diff:
        parts.append('<span class="diff-none">（無差異）</span>')
    parts.append('</div>')
    return ''.join(parts)


def build_diff_md(snapshots: list[dict], current_index: int) -> str:
    """Human-readable diff process for note/diff.md (versions up to the current one)."""
    lines = ['# 今日病歷 diff 過程', '']
    if not snapshots:
        return '\n'.join(lines + ['（無版本）', ''])
    lines.append(f'共 {len(snapshots)} 個版本；目前停留版本：{current_index + 1}。')
    lines.append('')
    for index, snap in enumerate(snapshots):
        meta = snap.get('meta') or {}
        label = f'## 版本 {index + 1} · 來源：{snap.get("source", "?")} · {snap.get("timestamp", "")}'
        if snap.get('t') is not None:
            label += f' · t+{format_clock(snap["t"])}'
        if index == current_index:
            label += ' · ◀ 目前'
        lines.append(label)
        if meta.get('job_id'):
            lines.append(f'- 作業：{meta["job_id"]}')
        review = meta.get('review')
        if isinstance(review, dict):
            if review.get('skipped'):
                lines.append('- 審查：未審查（n=0 對照組）')
            else:
                lines.append(f'- 審查：{review.get("rounds", "?")} 輪、通過 {review.get("passes", "?")} 次')
        if meta.get('call_ids'):
            lines.append(f'- LLM 呼叫：{", ".join(meta["call_ids"])}')
        lines.append('')
        if index == 0:
            lines += ['```text', snap.get('note', ''), '```', '']
            continue
        diff = unified_diff(snapshots[index - 1].get('note', ''), snap.get('note', ''),
                            f'版本 {index}', f'版本 {index + 1}')
        lines += ['```diff', diff or '（無差異）', '```', '']
    return '\n'.join(lines)
