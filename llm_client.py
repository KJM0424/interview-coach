"""LLM API 호출 레이어.

앱은 complete()만 부르고, 어느 회사 API인지는 여기서만 다룬다.
SDK 예외는 LLMError 계열로 바꿔서 올려 보낸다.
"""

from __future__ import annotations

from typing import Any, Protocol

import anthropic
import httpx
import httpx2
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

PROVIDERS = ("gemini", "anthropic")
DEFAULT_PROVIDER = "gemini"
MODELS = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-haiku-5-5",
}
API_KEY_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
MAX_TOKENS = 16000
# SDK 자동 재시도 횟수 (첫 호출 제외). 429, 5xx, 연결 오류에서 다시 시도한다.
MAX_RETRIES = 2


class LLMError(Exception):
    """API 호출 실패. 메시지는 화면에 그대로 보여 줄 수 있게 쓴다."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMResponseError(LLMError):
    """응답을 쓸 수 없고 다시 보내도 같을 가능성이 큰 경우 (거절, 안전 필터 차단 등)."""


class TruncatedResponseError(ValueError):
    """응답이 잘리거나 비었다. JSON 파싱 실패와 같이 취급해 다시 시도한다."""


class LLMClient(Protocol):
    def complete(self, system: str, user_prompt: str, json_mode: bool = False) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str, sdk: Any = None) -> None:
        self.sdk = sdk or anthropic.Anthropic(api_key=api_key, max_retries=MAX_RETRIES)

    def complete(self, system: str, user_prompt: str, json_mode: bool = False) -> str:
        try:
            response = self.sdk.messages.create(
                model=MODELS["anthropic"],
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.AuthenticationError as error:
            raise LLMAuthError("Anthropic API 키가 올바르지 않아요.") from error
        except anthropic.RateLimitError as error:
            raise LLMRateLimitError("요청이 너무 많아요. 잠시 후 다시 시도해 주세요.") from error
        except anthropic.APIStatusError as error:
            raise LLMError(f"API 오류가 났어요 ({error.status_code}): {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise LLMConnectionError(
                "API 서버에 연결하지 못했어요. 네트워크를 확인해 주세요."
            ) from error

        if response.stop_reason == "refusal":
            raise LLMResponseError("모델이 요청에 응답하지 않았어요. 입력 내용을 확인해 주세요.")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            raise TruncatedResponseError("응답이 최대 길이에서 잘렸어요.")
        if not text:
            raise TruncatedResponseError("응답에 텍스트가 없어요.")
        return text


class GeminiClient:
    TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)
    BLOCKED_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

    def __init__(self, api_key: str, sdk: Any = None) -> None:
        self.sdk = sdk or genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=MAX_RETRIES + 1)
            ),
        )

    def complete(self, system: str, user_prompt: str, json_mode: bool = False) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_TOKENS,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            response = self.sdk.models.generate_content(
                model=MODELS["gemini"], contents=user_prompt, config=config
            )
        except genai_errors.APIError as error:
            raise self._convert_api_error(error) from error
        except self.TRANSPORT_ERRORS as error:
            raise LLMConnectionError(
                "API 서버에 연결하지 못했어요. 네트워크를 확인해 주세요."
            ) from error

        feedback = response.prompt_feedback
        if feedback and feedback.block_reason:
            raise LLMResponseError("입력 내용이 Gemini 안전 필터에 막혔어요. 내용을 확인해 주세요.")
        finish = self._finish_reason(response)
        if finish in self.BLOCKED_REASONS:
            raise LLMResponseError("응답이 Gemini 안전 필터에 막혔어요. 다시 시도해 주세요.")
        text = (response.text or "").strip()
        if finish == "MAX_TOKENS":
            raise TruncatedResponseError("응답이 최대 길이에서 잘렸어요.")
        if not text:
            raise TruncatedResponseError("응답에 텍스트가 없어요.")
        return text

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        if not response.candidates:
            return None
        reason = response.candidates[0].finish_reason
        return getattr(reason, "name", reason)

    @staticmethod
    def _convert_api_error(error: genai_errors.APIError) -> LLMError:
        message = error.message or str(error)
        # Gemini는 잘못된 키에 400 INVALID_ARGUMENT를 돌려주기도 한다
        if error.code in (401, 403) or "API key" in message:
            return LLMAuthError("Gemini API 키가 올바르지 않아요.")
        if error.code == 429:
            return LLMRateLimitError(
                "Gemini 무료 사용 한도에 걸렸어요. 1분쯤 기다렸다가 다시 시도해 주세요."
            )
        return LLMError(f"API 오류가 났어요 ({error.code}): {message}")


def create_client(provider: str, api_key: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(api_key)
    if provider == "gemini":
        return GeminiClient(api_key)
    raise ValueError(
        f"지원하지 않는 LLM_PROVIDER입니다: {provider} ({', '.join(PROVIDERS)} 중 선택)"
    )
