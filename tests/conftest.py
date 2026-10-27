"""테스트 공통 설정. 실제 API 대신 FakeClient를 쓴다."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prompts  # noqa: E402

TEST_DATA = ROOT / "test_data"
APP_PATH = str(ROOT / "app.py")


def make_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    """thinking 블록이 앞에 붙은 Anthropic Messages API 응답 모양을 만든다."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


def evaluation_json(score: float, improvement: str = "개선 | 포인트") -> str:
    return json.dumps(
        {
            "evaluations": [
                {
                    "evaluator": e["name"],
                    "score": score,
                    "reason": f"{e['name']} 이유",
                    "improvement": improvement,
                }
                for e in prompts.EVALUATORS
            ],
            "gap": "구체 사례 부족",
        },
        ensure_ascii=False,
    )


class FakeClient:
    """llm_client.LLMClient와 같은 모양.

    handler(system, user)가 응답 텍스트를 돌려주거나 예외를 낸다.
    """

    def __init__(self, handler: Callable[[str, str], str]):
        self.handler = handler
        self.calls: list[tuple[str, bool]] = []

    def complete(self, system: str, user_prompt: str, json_mode: bool = False) -> str:
        self.calls.append((system, json_mode))
        return self.handler(system, user_prompt)


class InterviewScript:
    """면접 흐름용 가짜 응답. 답변에 'good'이 있으면 8점, 없으면 5점을 준다."""

    def __init__(self) -> None:
        self.bad_json_next = 0
        self.calls: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append(system)
        if system == prompts.QUESTION_SYSTEM:
            number = user.split("<진행>")[1].split("번째")[0]
            return f'"질문{number}: 프로젝트에서 맡은 역할은?"'
        if system == prompts.FOLLOWUP_SYSTEM:
            return "꼬리: 그 수치는 어떻게 측정했나요?"
        if system == prompts.EVALUATION_SYSTEM:
            if self.bad_json_next > 0:
                self.bad_json_next -= 1
                return "평가 결과는 다음과 같습니다 {invalid json"
            answer = user.split("<답변>")[1]
            score = 8 if "good" in answer else 5
            return "```json\n" + evaluation_json(score) + "\n```"
        if system == prompts.REPORT_SYSTEM:
            return json.dumps(
                {
                    "strengths": ["s1", "s2", "s3", "s4"],
                    "weaknesses": ["w1", "w2", "w3"],
                    "model_answer": "모범",
                },
                ensure_ascii=False,
            )
        raise AssertionError("알 수 없는 시스템 프롬프트")


@pytest.fixture
def script() -> InterviewScript:
    return InterviewScript()
