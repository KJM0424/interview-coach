"""LLM 호출 테스트: API별 클라이언트(llm_client.py)와 앱의 JSON 파싱·재시도·검증."""

import json
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest
from conftest import FakeClient, evaluation_json, make_response
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import app
import llm_client
import prompts

# --- Gemini 클라이언트 ---


def gemini_response(*parts, finish="STOP", block_reason=None):
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=list(parts)),
                finish_reason=finish,
            )
        ],
        prompt_feedback=(
            genai_types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
            if block_reason
            else None
        ),
    )


class FakeGeminiSDK:
    def __init__(self, result):
        self.result = result
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def gemini_client(result):
    sdk = FakeGeminiSDK(result)
    return llm_client.GeminiClient("key", sdk=sdk), sdk


def test_gemini_sends_system_model_and_json_mode():
    client, sdk = gemini_client(gemini_response(genai_types.Part(text='{"a": 1}')))
    assert client.complete("시스템", "사용자", json_mode=True) == '{"a": 1}'
    request = sdk.requests[0]
    assert request["model"] == "gemini-3.8-flash"
    assert request["contents"] == "사용자"
    assert request["config"].system_instruction == "시스템"
    assert request["config"].response_mime_type == "application/json"


def test_gemini_text_mode_has_no_json_mime_type():
    client, sdk = gemini_client(gemini_response(genai_types.Part(text="질문")))
    client.complete("s", "u")
    assert sdk.requests[0]["config"].response_mime_type is None


def test_gemini_skips_thought_parts():
    client, _ = gemini_client(
        gemini_response(
            genai_types.Part(text="생각 과정", thought=True),
            genai_types.Part(text="  최종 답  "),
        )
    )
    assert client.complete("s", "u") == "최종 답"


def test_gemini_max_tokens_is_retryable_truncation():
    client, _ = gemini_client(gemini_response(genai_types.Part(text='{"a":'), finish="MAX_TOKENS"))
    with pytest.raises(llm_client.TruncatedResponseError):
        client.complete("s", "u")


@pytest.mark.parametrize(
    "response",
    [
        gemini_response(genai_types.Part(text="x"), finish="SAFETY"),
        gemini_response(genai_types.Part(text="x"), block_reason="SAFETY"),
    ],
)
def test_gemini_safety_block_is_not_retried(response):
    client, _ = gemini_client(response)
    with pytest.raises(llm_client.LLMResponseError, match="안전 필터"):
        client.complete("s", "u")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            genai_errors.ClientError(
                400,
                {
                    "error": {
                        "code": 400,
                        "message": "API key not valid.",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            ),
            llm_client.LLMAuthError,
        ),
        (
            genai_errors.ClientError(403, {"error": {"code": 403, "message": "denied"}}),
            llm_client.LLMAuthError,
        ),
        (
            genai_errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "message": "Resource exhausted",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
            ),
            llm_client.LLMRateLimitError,
        ),
        (
            genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}}),
            llm_client.LLMError,
        ),
        (httpx.ConnectError("down"), llm_client.LLMConnectionError),
        (httpx2.ConnectError("down"), llm_client.LLMConnectionError),
    ],
)
def test_gemini_errors_are_converted(error, expected):
    client, _ = gemini_client(error)
    with pytest.raises(expected) as caught:
        client.complete("s", "u")
    assert type(caught.value) is expected


def test_gemini_rate_limit_message_mentions_free_quota():
    error = genai_errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted"}})
    client, _ = gemini_client(error)
    with pytest.raises(llm_client.LLMRateLimitError, match="무료 사용 한도"):
        client.complete("s", "u")


# --- Anthropic 클라이언트 ---


def anthropic_client(result):
    def create(**kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm_client.AnthropicClient("key", sdk=sdk)


def anthropic_status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(status, request=request), body=None)


def test_anthropic_ignores_thinking_blocks():
    assert anthropic_client(make_response("  질문입니다  ")).complete("s", "u") == "질문입니다"


def test_anthropic_refusal_is_not_retried():
    with pytest.raises(llm_client.LLMResponseError):
        anthropic_client(make_response("", stop_reason="refusal")).complete("s", "u")


def test_anthropic_max_tokens_is_retryable_truncation():
    with pytest.raises(llm_client.TruncatedResponseError):
        anthropic_client(make_response("{", stop_reason="max_tokens")).complete("s", "u")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic_status_error(anthropic.AuthenticationError, 401), llm_client.LLMAuthError),
        (anthropic_status_error(anthropic.RateLimitError, 429), llm_client.LLMRateLimitError),
        (anthropic_status_error(anthropic.InternalServerError, 500), llm_client.LLMError),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            llm_client.LLMConnectionError,
        ),
    ],
)
def test_anthropic_errors_are_converted(error, expected):
    with pytest.raises(expected) as caught:
        anthropic_client(error).complete("s", "u")
    assert type(caught.value) is expected


def test_create_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        llm_client.create_client("openai", "key")


# --- 앱의 JSON 처리 ---


def responses(*items):
    """차례대로 텍스트를 돌려주거나 예외를 내는 handler."""
    queue = list(items)

    def handler(system, user):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def test_json_calls_use_json_mode():
    client = FakeClient(responses(evaluation_json(8)))
    app.call_llm_json(client, "s", "u", app.validate_evaluation)
    assert client.calls == [("s", True)]


def test_truncated_response_is_retried():
    client = FakeClient(responses(llm_client.TruncatedResponseError("잘림"), evaluation_json(8)))
    result = app.call_llm_json(client, "s", "u", app.validate_evaluation)
    assert result["average"] == 8
    assert len(client.calls) == 2


def test_json_parse_failure_retries_once_then_succeeds():
    client = FakeClient(responses("JSON이 아닌 답변", evaluation_json(6)))
    assert app.call_llm_json(client, "s", "u", app.validate_evaluation)["average"] == 6
    assert len(client.calls) == 2


def test_json_parse_failure_twice_raises():
    client = FakeClient(responses("{ broken", "{ broken"))
    with pytest.raises(llm_client.LLMResponseError, match="2번 실패"):
        app.call_llm_json(client, "s", "u", app.validate_evaluation)
    assert len(client.calls) == 2


def test_blocked_response_is_not_retried():
    client = FakeClient(responses(llm_client.LLMResponseError("막힘"), evaluation_json(8)))
    with pytest.raises(llm_client.LLMResponseError, match="막힘"):
        app.call_llm_json(client, "s", "u", app.validate_evaluation)
    assert len(client.calls) == 1


def test_text_call_turns_truncation_into_response_error():
    client = FakeClient(responses(llm_client.TruncatedResponseError("잘림")))
    with pytest.raises(llm_client.LLMResponseError):
        app.call_llm_text(client, "s", "u")
    assert client.calls == [("s", False)]


def test_parse_json_object_handles_code_fence_and_preamble():
    text = '평가입니다.\n```json\n{"a": 1}\n```'
    assert app.parse_json_object(text) == {"a": 1}


def test_validate_evaluation_computes_average_in_code():
    data = json.loads(evaluation_json(7))
    data["evaluations"][0]["score"] = 10
    result = app.validate_evaluation(data)
    assert result["average"] == 7.8  # (10 + 7 + 7 + 7) / 4 = 7.75 → 7.8
    assert [e["evaluator"] for e in result["evaluations"]] == [
        e["name"] for e in prompts.EVALUATORS
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(evaluations=[]),
        lambda d: d.update(evaluations="not a list"),
        lambda d: d["evaluations"][0].update(score=11),
        lambda d: d["evaluations"][0].update(score=-1),
        lambda d: d["evaluations"][0].update(score=None),
        lambda d: d["evaluations"][0].update(score="여덟"),
        lambda d: d["evaluations"][1].update(reason=" "),
        lambda d: d["evaluations"][2].update(improvement=""),
        lambda d: d["evaluations"][3].update(evaluator="사장"),
        lambda d: d["evaluations"].pop(),
    ],
)
def test_validate_evaluation_rejects_bad_data(mutate):
    data = json.loads(evaluation_json(7))
    mutate(data)
    with pytest.raises((ValueError, KeyError, TypeError)):
        app.validate_evaluation(data)


def test_validate_report_trims_to_three_items():
    result = app.validate_report(
        {"strengths": ["a", "b", "c", "d"], "weaknesses": ["e", "f", "g"], "model_answer": "m"}
    )
    assert result["strengths"] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "data",
    [
        {"strengths": "abc", "weaknesses": ["e", "f", "g"], "model_answer": "m"},
        {"strengths": ["a", "b"], "weaknesses": ["e", "f", "g"], "model_answer": "m"},
        {"strengths": ["a", "b", "c"], "weaknesses": ["e", "f", "g"], "model_answer": " "},
    ],
)
def test_validate_report_rejects_bad_data(data):
    with pytest.raises((ValueError, TypeError)):
        app.validate_report(data)


def test_clean_question_strips_quotes():
    assert app.clean_question(' "질문입니다?" ') == "질문입니다?"
