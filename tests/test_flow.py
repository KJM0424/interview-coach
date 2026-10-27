"""Streamlit AppTest로 면접 전체 흐름을 확인한다. 실제 API는 부르지 않는다."""

from unittest import mock

import pytest
import streamlit as st
from conftest import APP_PATH, FakeClient
from streamlit.testing.v1 import AppTest


def isolate_settings(monkeypatch, **env):
    """실제 .env와 환경변수를 읽지 않게 하고, 필요한 값만 넣는다."""
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    st.cache_resource.clear()


@pytest.fixture
def run_app(script, monkeypatch):
    """FakeClient를 쓰는 AppTest를 만든다 (기본 공급자 gemini)."""
    isolate_settings(monkeypatch, GEMINI_API_KEY="test-key")
    created = []

    def fake_create_client(provider, api_key):
        created.append((provider, api_key))
        return FakeClient(script)

    monkeypatch.setattr("llm_client.create_client", fake_create_client)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert created == [("gemini", "test-key")]
    return at


def html(at) -> str:
    return "\n".join(e.proto.body for e in at.get("html"))


def last(at) -> dict:
    return at.session_state.messages[-1]


def answer(at, text: str) -> None:
    at.chat_input[0].set_value(text).run()
    assert not at.exception


def test_full_interview_flow(run_app, script):
    at = run_app
    assert at.session_state.stage == "setup"
    assert "실전 모의 면접" in html(at)
    assert "실제 개인 정보는 넣지 마세요" in html(at)

    at.button(key="start").click().run()
    assert at.session_state.stage == "interview"
    assert last(at)["label"] == "질문 1 / 5"
    assert '"' not in last(at)["text"]
    assert at.chat_input[0].placeholder == "답변을 입력하거나 마이크를 눌러 말해 주세요"
    assert "webkitSpeechRecognition" in html(at), "음성 입력 스크립트가 들어가야 함"
    assert "Gemini API로 채점 중이에요" in html(at)

    # 5점 답변 3번: 재답변 1/2, 2/2, 그다음 다음 질문
    answer(at, "bad 1")
    assert last(at)["kind"] == "reanswer" and last(at)["attempt"] == 1
    assert "재답변 1 / 2" in at.chat_input[0].placeholder
    assert "기준 미달" in html(at)
    answer(at, "bad 2")
    assert last(at)["attempt"] == 2
    answer(at, "bad 3")
    assert last(at)["label"] == "질문 2 / 5"
    assert at.session_state.messages[-2]["kind"] == "notice"
    assert at.session_state.records[0]["followup"] is None

    # JSON 1회 실패 → 재시도 성공 → 꼬리 질문
    script.bad_json_next = 1
    answer(at, "good answer")
    assert last(at)["label"] == "꼬리 질문"
    assert at.chat_input[0].placeholder == "꼬리 질문에 답해 주세요"

    # JSON 2회 실패 → 에러, 상태 그대로, 같은 답변으로 다시 시도
    script.bad_json_next = 2
    before = len(at.session_state.messages)
    answer(at, "good followup")
    assert "2번 실패" in at.error[0].value
    assert len(at.session_state.messages) == before
    assert at.session_state.pending_answer == "good followup"
    next(b for b in at.button if b.label == "같은 답변으로 다시 시도").click().run()
    assert at.session_state.pending_answer is None
    assert last(at)["label"] == "질문 3 / 5"

    for _ in range(3):
        answer(at, "good main")
        answer(at, "good tail")

    assert at.session_state.stage == "report"
    report = at.session_state.report
    assert report["strengths"] == ["s1", "s2", "s3"]
    assert report["target"]["label"] == "Q1"

    body = html(at)
    # 채점 답변: Q1(5점) 1개 + Q2~Q5 본/꼬리(8점) 8개 → 평가위원 평균 (5 + 8*8) / 9 = 7.7
    assert "평균 <em>7.7점</em>이에요" in body
    assert body.count('<p class="stat-value">7.7<span>/ 10</span></p>') == 4
    assert "답변 3회 · 기술 5 · 인사 5 · 실무 5 · 임원 5" in body
    assert body.count('<span class="chip">') == 9
    assert at.get("download_button")
    # 질문 생성 5 + 평가 15(재시도 포함) + 꼬리 질문 4 + 리포트 1
    assert len(script.calls) == 24


def test_report_markdown_for_download(run_app):
    at = run_app
    at.button(key="start").click().run()
    for _ in range(5):
        answer(at, "good main")
        answer(at, "good tail")
    assert at.session_state.stage == "report"

    import app

    state = mock.Mock(records=at.session_state.records, context=at.session_state.context)
    with mock.patch.object(app.st, "session_state", state):
        markdown = app.build_report_markdown(at.session_state.report)
    assert "## 질문별 점수표" in markdown and "## 평가위원별 평균" in markdown
    assert "| Q1 | 질문1: 프로젝트에서 맡은 역할은? | 1 | 8 | 8 | 8 | 8 | 8 |" in markdown
    assert "| Q1-꼬리 |" in markdown


def test_restart_keeps_previous_inputs(run_app):
    at = run_app
    at.segmented_control[0].set_value("직접 입력").run()
    at.text_input[0].set_value("데이터 엔지니어").run()
    at.text_area[0].set_value("결제 회사 인재상").run()
    at.button(key="start").click().run()
    assert at.session_state.context["role"] == "데이터 엔지니어"

    at.button(key="restart").click().run()
    assert at.session_state.stage == "setup"
    assert at.session_state.messages == []
    assert at.text_area[0].value == "결제 회사 인재상"
    assert at.text_input[0].value == "데이터 엔지니어"


def test_custom_role_is_required(run_app):
    at = run_app
    at.segmented_control[0].set_value("직접 입력").run()
    at.button(key="start").click().run()
    assert at.session_state.stage == "setup"
    assert "직군 이름을 입력해 주세요." in [e.value for e in at.error]


def test_missing_gemini_key_shows_guide(monkeypatch):
    isolate_settings(monkeypatch)
    at = AppTest.from_file(APP_PATH).run()
    message = at.error[0].value
    assert "GEMINI_API_KEY가 없어요" in message and "Google AI Studio" in message


def test_anthropic_provider_needs_anthropic_key_and_has_no_notice(monkeypatch, script):
    isolate_settings(monkeypatch, LLM_PROVIDER="anthropic")
    at = AppTest.from_file(APP_PATH).run()
    assert "ANTHROPIC_API_KEY가 없어요" in at.error[0].value

    isolate_settings(monkeypatch, LLM_PROVIDER="Anthropic", ANTHROPIC_API_KEY="a-key")
    created = []
    monkeypatch.setattr(
        "llm_client.create_client",
        lambda provider, api_key: created.append((provider, api_key)) or FakeClient(script),
    )
    at = AppTest.from_file(APP_PATH).run()
    assert created == [("anthropic", "a-key")]
    assert "실제 개인 정보는 넣지 마세요" not in html(at)


def test_unknown_provider_is_rejected(monkeypatch):
    isolate_settings(monkeypatch, LLM_PROVIDER="openai", GEMINI_API_KEY="k")
    at = AppTest.from_file(APP_PATH).run()
    assert "LLM_PROVIDER 값 'openai'를 알 수 없어요" in at.error[0].value


def test_secrets_take_priority_over_env(monkeypatch, script):
    isolate_settings(monkeypatch, GEMINI_API_KEY="from-env")
    seen = []
    monkeypatch.setattr(
        "llm_client.create_client",
        lambda provider, api_key: seen.append(api_key) or FakeClient(script),
    )
    at = AppTest.from_file(APP_PATH)
    at.secrets["GEMINI_API_KEY"] = "from-secrets"
    at.run()
    assert seen == ["from-secrets"]


def test_auth_error_tells_which_key_to_check(monkeypatch):
    import llm_client

    def handler(system, user):
        raise llm_client.LLMAuthError("Gemini API 키가 올바르지 않아요.")

    isolate_settings(monkeypatch, GEMINI_API_KEY="bad")
    monkeypatch.setattr("llm_client.create_client", lambda provider, api_key: FakeClient(handler))
    at = AppTest.from_file(APP_PATH).run()
    at.button(key="start").click().run()
    assert at.session_state.stage == "setup"
    assert at.error[0].value == (
        "Gemini API 키가 올바르지 않아요. "
        ".streamlit/secrets.toml 또는 .env의 GEMINI_API_KEY를 확인해 주세요."
    )
