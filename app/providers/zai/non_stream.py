#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 非流式响应聚合器

将上游 SSE 流聚合为单个非流式 OpenAI 格式响应。
"""

import json
import traceback
from typing import Dict, Any, AsyncGenerator

from app.providers.zai._chunk_utils import create_openai_response_with_reasoning
from app.utils.logger import get_logger

logger = get_logger()


def _clean_thinking_content(delta_content: str) -> str:
    if delta_content.startswith("<details") and "</summary>\n>" in delta_content:
        return delta_content.split("</summary>\n>")[-1].strip()
    return delta_content


async def aggregate_non_stream_response(
    lines: AsyncGenerator[str, None],
    chat_id: str,
    model: str,
) -> Dict[str, Any]:
    final_content = ""
    reasoning_content = ""
    usage_info: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    cleaned = ""

    try:
        async for line in lines:
            if not line or not line.strip().startswith("data:"):
                continue

            data_str = line.strip()[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("type") != "chat:completion":
                continue

            data = chunk.get("data", {})
            phase = data.get("phase")
            delta_content = data.get("delta_content", "")
            edit_content = data.get("edit_content", "")

            if data.get("usage"):
                usage_info = data["usage"]

            if phase == "thinking":
                if delta_content:
                    cleaned = _clean_thinking_content(delta_content)
                    reasoning_content += cleaned
            elif phase == "answer":
                if edit_content and "</details>" in edit_content:
                    reasoning_content += edit_content.split(cleaned)[-1].replace(
                        "</details>", ""
                    )
                elif delta_content:
                    final_content += delta_content
            elif phase == "other":
                if edit_content:
                    final_content += edit_content

    except Exception as e:
        logger.error("非流式响应处理错误: {}", e)
        logger.error(traceback.format_exc())
        return {
            "error": {
                "message": f"非流式聚合 错误: {str(e)}",
                "type": "provider_error",
                "code": "internal_error",
            }
        }

    final_content = (final_content or "").strip()
    reasoning_content = (reasoning_content or "").strip()
    if not final_content and reasoning_content:
        final_content = reasoning_content

    return create_openai_response_with_reasoning(
        chat_id, model, final_content, reasoning_content, usage_info
    )
