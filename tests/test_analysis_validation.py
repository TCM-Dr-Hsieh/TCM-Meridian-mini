"""整體分析的輸出驗證：A&T 要涵蓋分析模板的每個項目、互評要有四個標題、仲裁的最終 A&T 同樣要涵蓋模板項目。"""
import pytest

from mini.agents.analysis_job import CROSS_HEADINGS, check_sections, parse_arbitration, template_sections
from mini.llm import ValidationError

TEMPLATE = ('\t一- 西醫診斷：\n\t二- 中醫診斷：\n\t三- 中醫病機/證型：\n\t四- 中醫治則：\n'
            '\t五- 處方(中藥或針灸)：\n\t六- 衛教/建議轉診：\n')
KEYS = ['西醫診斷', '中醫診斷', '中醫病機', '中醫治則', '處方', '衛教']
FULL = ('## 一- 西醫診斷\n- 偏頭痛（待確認）\n## 二- 中醫診斷\n- 頭痛\n## 三- 中醫病機/證型\n- 肝陽上亢\n'
        '## 四- 中醫治則\n- 平肝潛陽\n## 五- 處方(中藥或針灸)\n- 建議由醫師決定\n## 六- 衛教/建議轉診\n- 規律作息')


# --- template parsing ---------------------------------------------------------
def test_template_sections_use_the_leading_words_of_each_title():
    assert template_sections(TEMPLATE) == KEYS


def test_free_form_or_blank_templates_have_no_required_sections():
    assert template_sections('') == []
    assert template_sections('請自由發揮，不必分項。') == []


def test_only_top_level_numbered_items_are_required():
    template = '1. 主訴\n    1. 子項不必單獨出現\n2. 處置'
    assert template_sections(template) == ['主訴', '處置']


# --- section checking ---------------------------------------------------------
def test_a_complete_at_passes_and_free_form_templates_accept_anything():
    assert check_sections(FULL, KEYS, '項目') == FULL
    assert check_sections('隨便寫的一段話', [], '項目') == '隨便寫的一段話'


@pytest.mark.parametrize('heading_style', [
    '### 1. 西醫診斷：偏頭痛', '**一、西醫診斷**\n偏頭痛', '（一）西醫診斷：偏頭痛', '一- 西醫診斷：偏頭痛', '西醫診斷：偏頭痛',
    '- **西醫診斷**：偏頭痛',
])
def test_common_heading_styles_are_recognised(heading_style):
    check_sections(heading_style, ['西醫診斷'], '項目')


def test_slash_titles_match_on_their_first_words():
    check_sections('六- 衛教／轉診\n- 規律作息', ['衛教'], '項目')
    check_sections('## 衛教\n- 規律作息', ['衛教'], '項目')


def test_missing_items_are_named_in_the_error():
    broken = FULL.replace('## 四- 中醫治則\n- 平肝潛陽\n', '').replace('## 五- 處方(中藥或針灸)\n- 建議由醫師決定\n', '')
    with pytest.raises(ValidationError) as error:
        check_sections(broken, KEYS, '分析模板項目')
    assert '中醫治則' in str(error.value) and '處方' in str(error.value) and '西醫診斷' not in str(error.value)


def test_an_item_with_no_content_is_rejected():
    empty_body = FULL.replace('- 平肝潛陽', '')
    with pytest.raises(ValidationError, match='沒有內容.*中醫治則'):
        check_sections(empty_body, KEYS, '分析模板項目')


def test_a_parenthetical_in_the_title_does_not_count_as_content():
    text = FULL.replace('- 建議由醫師決定', '')
    with pytest.raises(ValidationError, match='沒有內容.*處方'):
        check_sections(text, KEYS, '分析模板項目')


def test_inline_content_after_a_colon_counts_as_content():
    text = FULL.replace('## 六- 衛教/建議轉診\n- 規律作息', '## 六- 衛教/建議轉診：規律作息')
    check_sections(text, KEYS, '分析模板項目')


def test_information_insufficient_is_valid_content():
    check_sections(FULL.replace('- 平肝潛陽', '- 資訊不足，需補問舌脈'), KEYS, '分析模板項目')


def test_a_title_that_starts_with_a_numeral_character_is_not_eaten_by_numbering():
    check_sections('三- 一般檢查：\n- 血壓正常', ['一般檢查'], '項目')
    check_sections('一般檢查：血壓正常', ['一般檢查'], '項目')


def test_cross_review_needs_all_four_headings():
    complete = ('## 對方版本的優點\n- 好\n## 對方版本的缺點與風險\n- 無\n## 與我方版本的分歧\n- 實質一致\n'
                '## 建議採納與不建議採納\n- 建議採納')
    check_sections(complete, CROSS_HEADINGS, '評比標題')
    with pytest.raises(ValidationError, match='與我方版本的分歧'):
        check_sections(complete.replace('## 與我方版本的分歧\n- 實質一致\n', ''), CROSS_HEADINGS, '評比標題')
    with pytest.raises(ValidationError, match='對方版本的優點'):
        check_sections('我覺得對方寫得不錯，但是處方有點問題，建議再討論一下細節。', CROSS_HEADINGS, '評比標題')


# --- arbitration --------------------------------------------------------------
def arbitration(final: str) -> str:
    return f'===ARBITRATION===\n意見一致。\n===FINAL_AT===\n{final}'


def test_arbitration_final_at_must_cover_the_template():
    notes, final = parse_arbitration(arbitration(FULL), KEYS)
    assert notes == '意見一致。' and final == FULL
    with pytest.raises(ValidationError, match='最終 A&T.*中醫病機'):
        parse_arbitration(arbitration(FULL.replace('## 三- 中醫病機/證型\n- 肝陽上亢\n', '')), KEYS)


def test_arbitration_without_keys_only_checks_markers_and_length():
    assert parse_arbitration(arbitration('一段夠長的自由格式內容，完全沒有任何分析模板的項目與編號。'))[1].startswith('一段')
    with pytest.raises(ValidationError, match='標記'):
        parse_arbitration('沒有標記' * 10)
