"""화면용 HTML 조각 테스트. 사용자·모델 텍스트가 이스케이프되는지 본다."""

import json
import re

from conftest import evaluation_json

import app

ATTACK = '<script>alert(1)</script><img src=x onerror="alert(2)">'


def test_answer_html_escapes_tags_and_keeps_line_breaks():
    html = app.message_html({"role": "user", "kind": "answer", "text": f"첫 줄\n{ATTACK}"})
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert "첫 줄<br>" in html


def test_evaluation_html_escapes_model_text():
    evaluation = app.validate_evaluation(json.loads(evaluation_json(5, improvement=ATTACK)))
    html = app.evaluation_html("답변 평가", evaluation)
    assert "<script>" not in html and "<img" not in html
    assert "기준 미달" in html and 'class="bar low"' in html


def test_evaluation_html_marks_pass():
    evaluation = app.validate_evaluation(json.loads(evaluation_json(8)))
    html = app.evaluation_html("답변 평가", evaluation)
    assert "기준 통과" in html and 'class="bar low"' not in html


def test_question_and_reanswer_html_escape():
    assert "<script>" not in app.question_html("질문 1 / 5", ATTACK)
    evaluation = app.validate_evaluation(json.loads(evaluation_json(5)))
    evaluation["gap"] = ATTACK
    html = app.message_html(
        {
            "role": "assistant",
            "kind": "reanswer",
            "evaluation": evaluation,
            "question": ATTACK,
            "attempt": 1,
        }
    )
    assert "<script>" not in html and "1 / 2" in html


def test_score_bar_width_is_clamped():
    assert 'style="width:100%"' in app.score_bar(12)
    assert 'style="width:0%"' in app.score_bar(-3)


def test_voice_script_has_no_html_tag_strings():
    """st.html은 스크립트 안에 태그 모양 문자열이 있으면 스크립트를 실행하지 않는다."""
    script = app.VOICE_SCRIPT_PATH.read_text(encoding="utf-8")
    assert not re.search(r"<[a-zA-Z/!]", script)
