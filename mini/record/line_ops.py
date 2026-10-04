"""Line-level edit engine (ported from TCM-Meridian Record_Subagent.apply_operations).

All `line` numbers refer to the ORIGINAL text shown to the model, so earlier operations
never shift later ones. The whole batch is validated first and rebuilt once; any invalid
operation rejects the entire batch.
"""
from __future__ import annotations

from typing import Any


class OperationsError(ValueError):
    pass


def split_lines(text: str) -> list[str]:
    return text.split('\n') if text else ['']


def number_lines(text: str) -> str:
    """Render text with 1-based line numbers for the model."""
    if not text:
        return '（空白，尚無任何內容；請用 insert，line 設為 1）'
    return '\n'.join(f'{i + 1:3d} | {line}' for i, line in enumerate(split_lines(text)))


def parse_operations(data: Any) -> list[dict]:
    """Validate the shape of the model's JSON ({"operations": [...]}) and return the list."""
    if not isinstance(data, dict) or not isinstance(data.get('operations'), list):
        raise OperationsError('輸出必須是含 "operations" 陣列的 JSON 物件。')
    operations = data['operations']
    for index, item in enumerate(operations, 1):
        if not isinstance(item, dict):
            raise OperationsError(f'第{index}個操作不是 JSON 物件。')
        if item.get('op') in ('insert', 'replace') and not isinstance(item.get('content'), str):
            raise OperationsError(f'第{index}個操作 {item.get("op")} 缺少字串 content。')
        if isinstance(item.get('content'), str) and '\n' in item['content']:
            raise OperationsError(f'第{index}個操作 content 不可含換行；每個操作只處理一行。')
    return operations


def apply_operations(text: str, operations: list[dict]) -> tuple[bool, str, list[str], list[str]]:
    """Apply insert/delete/replace operations.

    Returns (ok, new_text, logs, failures); when ok is False, new_text is the original text.
    """
    lines = split_lines(text)
    n = len(lines)
    logs: list[str] = []
    failures: list[str] = []
    normalized_ops: list[dict] = []
    inplace_targets: dict[int, str] = {}

    for i, op_data in enumerate(operations, 1):
        op = op_data.get('op', '')
        line_num = op_data.get('line', 0)
        normalized = dict(op_data)
        if op == 'insert':
            if isinstance(line_num, bool) or not isinstance(line_num, int) or line_num < 1:
                failures.append(f'第{i}個操作 insert：行號 {line_num} 超出可插入範圍 1~{n + 1}（原文共 {n} 行）')
            elif line_num > n + 1:
                normalized['line'] = n + 1
                logs.append(f'[NORMALIZE] 第{i}個 insert 行號 {line_num} 超出原文範圍，視為文末追加 line={n + 1}')
        elif op in ('delete', 'replace'):
            if isinstance(line_num, bool) or not isinstance(line_num, int) or not (1 <= line_num <= n):
                failures.append(f'第{i}個操作 {op}：行號 {line_num} 超出範圍 1~{n}（原文共 {n} 行）')
            elif line_num in inplace_targets:
                failures.append(f'第{i}個操作 {op}：行 {line_num} 已被 {inplace_targets[line_num]} 指向，'
                                f'同一行不可重複 delete/replace')
            else:
                inplace_targets[line_num] = op
        else:
            failures.append(f'第{i}個操作：未知 op「{op}」')
        normalized_ops.append(normalized)

    if failures:
        return False, text, logs, failures

    inserts_before: dict[int, list[str]] = {}
    replaced: dict[int, str] = {}
    deleted: set[int] = set()
    for op_data in normalized_ops:
        op, line_num, content = op_data['op'], op_data['line'], op_data.get('content', '')
        if op == 'insert':
            inserts_before.setdefault(line_num, []).append(content)
        elif op == 'replace':
            replaced[line_num] = content
        else:
            deleted.add(line_num)

    rebuilt: list[str] = []
    if text == '' and set(inserts_before).issubset({1, n + 1}) and not replaced and not deleted:
        # An empty document is one blank line: inserts fill it instead of leaving a stray blank.
        for ins in inserts_before.get(1, []) + inserts_before.get(2, []):
            rebuilt.append(ins)
            logs.append(f'[INSERT] 空白文件: "{ins}"')
    else:
        for line_num in range(1, n + 1):
            for ins in inserts_before.get(line_num, []):
                rebuilt.append(ins)
                logs.append(f'[INSERT] 行{line_num}前: "{ins}"')
            if line_num in deleted:
                logs.append(f'[DELETE] 行{line_num}: "{lines[line_num - 1]}"')
                continue
            if line_num in replaced:
                logs.append(f'[REPLACE] 行{line_num}: "{lines[line_num - 1]}" → "{replaced[line_num]}"')
                rebuilt.append(replaced[line_num])
            else:
                rebuilt.append(lines[line_num - 1])
        for ins in inserts_before.get(n + 1, []):
            rebuilt.append(ins)
            logs.append(f'[INSERT] 末尾: "{ins}"')
    return True, '\n'.join(rebuilt), logs, failures
