"""Small Amazon Bedrock Claude wrapper used by the live evaluation path."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


# Bedrock requires an inference profile for Claude Sonnet 4 in this account/region.
DEFAULT_BEDROCK_MODEL_ID = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_BEDROCK_REGION = "ap-southeast-2"


@dataclass(frozen=True)
class BedrockResponse:
    """Normalised response from Bedrock Converse."""

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class BedrockClaudeClient:
    """Calls Anthropic Claude through Amazon Bedrock's Converse API."""

    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        verify_ssl: bool | str = True,
        max_retries: int = 2,
    ) -> None:
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        self.profile_name = profile_name or os.environ.get("AWS_PROFILE") or None
        self.verify_ssl = os.environ.get("AWS_CA_BUNDLE") or verify_ssl
        self.max_retries = max_retries
        session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        config = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 2})
        self.client = session.client(
            "bedrock-runtime",
            region_name=self.region_name,
            config=config,
            verify=self.verify_ssl,
        )

    def converse_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], BedrockResponse]:
        """Ask Claude for a JSON object and parse it defensively."""
        last_error: Exception | None = None
        prompt = system_prompt
        for attempt in range(self.max_retries + 1):
            response = self.converse(
                system_prompt=prompt,
                user_text=json.dumps(payload, indent=2),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            try:
                return _extract_json_object(response.text), response
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                prompt = (
                    f"{system_prompt} Return strict JSON only. Escape all newlines and control characters "
                    "inside string values. Do not include markdown fences or explanatory text."
                )
                time.sleep((2**attempt) + 0.25)
        raise ValueError("Bedrock response did not contain valid JSON after retries") from last_error

    def converse(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> BedrockResponse:
        """Call Bedrock Converse and return text plus usage fields."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=[{"role": "user", "content": [{"text": user_text}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                )
                text = "".join(
                    block.get("text", "")
                    for block in raw.get("output", {}).get("message", {}).get("content", [])
                )
                usage = raw.get("usage", {})
                input_tokens = int(usage.get("inputTokens", 0))
                output_tokens = int(usage.get("outputTokens", 0))
                total_tokens = int(usage.get("totalTokens", input_tokens + output_tokens))
                return BedrockResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            except (BotoCoreError, ClientError) as exc:
                last_error = exc
                time.sleep((2**attempt) + 0.25)
        raise RuntimeError("Bedrock Converse call failed after retries") from last_error


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a model response that should contain exactly one JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Bedrock response did not contain JSON: {text!r}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object from Bedrock, received: {type(value).__name__}")
    return value
