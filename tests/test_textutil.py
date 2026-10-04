import pytest

from mini.record.line_ops import OperationsError, parse_operations
from mini.textutil import extract_json


def test_extract_json_handles_fences_think_blocks_and_surrounding_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert extract_json('<think>先想一想 {x}</think>{"a": 2}') == {'a': 2}
    assert extract_json('因此我傾向通過。\n\n{"pass": true, "issues": []}\n以上。') == {'pass': True, 'issues': []}


def test_extract_json_accepts_raw_tabs_and_newlines_inside_strings():
    # Models copy tab-indented template lines and multi-line reasoning verbatim into string values.
    text = '{"thinking": "L8: \t身高： 公分\n   L9: \t血壓： / mmHg", "operations": []}'
    data = extract_json(text)
    assert data['operations'] == [] and '\t身高' in data['thinking'] and '\n   L9' in data['thinking']
    wrapped = '先分析如下。\n{"pass": true,\n "comment": "第一行\n第二行\t完"}'
    assert extract_json(wrapped)['pass'] is True


def test_extract_json_still_rejects_real_syntax_errors_and_missing_objects():
    with pytest.raises(ValueError):
        extract_json('{"a": "x"\n "b": 2}')                       # missing comma: the retry has to fix this one
    with pytest.raises(ValueError, match='找不到'):
        extract_json('完全沒有 JSON')


def test_an_operation_content_with_a_raw_newline_is_still_rejected():
    data = extract_json('{"operations": [{"op": "insert", "line": 1, "content": "甲- 現病史\n乙- 過去病史"}]}')
    with pytest.raises(OperationsError, match='不可含換行'):
        parse_operations(data)
    tabbed = extract_json('{"operations": [{"op": "insert", "line": 1, "content": "\t身高： 公分"}]}')
    assert parse_operations(tabbed)[0]['content'] == '\t身高： 公分'            # a tab is legitimate template text
