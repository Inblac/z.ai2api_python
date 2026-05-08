#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI chunk 工具函数
"""

import json
import time
from typing import Dict, Any, Optional, AsyncGenerator


def create_openai_chunk(
    chat_id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    system_fingerprint: str = "fp_zai_001",
) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "system_fingerprint": system_fingerprint,
    }


def format_sse_chunk(chunk: Dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def create_openai_response_with_reasoning(
    chat_id: str,
    model: str,
    content: str,
    reasoning_content: str = None,
    usage: Optional[Dict[str, int]] = None,
    system_fingerprint: str = "fp_zai_001",
) -> Dict[str, Any]:
    message = {
        "role": "assistant",
        "content": content,
    }

    if reasoning_content and reasoning_content.strip():
        message["reasoning_content"] = reasoning_content

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": system_fingerprint,
    }


async def streaming_error_generator(
    error_msg: str, error_type: str, code: Optional[int] = None
) -> AsyncGenerator[str, None]:
    error_payload: Dict[str, Any] = {"message": error_msg, "type": error_type}
    if code is not None:
        error_payload["code"] = code
    error_response = {"error": error_payload}
    yield f"data: {json.dumps(error_response)}\n\n"
    yield "data: [DONE]\n\n"
