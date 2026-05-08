#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI SSE 非工具流解析器

处理 Z.AI SSE 流中的 thinking / answer / other / done 阶段，
生成 OpenAI 兼容的流式 chunk（不含工具调用，工具调用由 SSEToolHandler 处理）。
"""

import json
import traceback
from typing import AsyncGenerator

from app.providers.zai._chunk_utils import (
    create_openai_chunk,
    format_sse_chunk,
    streaming_error_generator,
)
from app.utils.logger import get_logger

logger = get_logger()


def _clean_thinking_content(delta_content: str) -> str:
    if delta_content.startswith("<details") and "</summary>\n>" in delta_content:
        return delta_content.split("</summary>\n>")[-1].strip()
    return delta_content


async def parse_non_tool_sse_stream(
    lines: AsyncGenerator[str, None],
    chat_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    logger.info("开始处理 SSE 流（非工具模式）")
    has_thinking = False
    delta_content = None
    buffer = ""
    line_count = 0
    last_phase = None

    try:
        async for line in lines:
            line_count += 1
            if not line:
                continue

            buffer += line + "\n"
            while "\n" in buffer:
                current_line, buffer = buffer.split("\n", 1)
                if not current_line.strip():
                    continue

                if not current_line.startswith("data:"):
                    continue

                chunk_str = current_line[5:].strip()
                if not chunk_str or chunk_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue

                logger.debug(
                    "解析数据块: {}",
                    chunk_str[:1000] + "..." if len(chunk_str) > 1000 else chunk_str,
                )

                try:
                    chunk = json.loads(chunk_str)
                    if chunk.get("type") != "chat:completion":
                        continue

                    data = chunk.get("data", {})
                    phase = data.get("phase")

                    if phase and phase != last_phase:
                        logger.info("SSE 阶段: {}", phase)
                        last_phase = phase

                    if phase == "thinking":
                        if not has_thinking:
                            has_thinking = True
                            role_chunk = create_openai_chunk(
                                chat_id, model, {"role": "assistant"}
                            )
                            yield format_sse_chunk(role_chunk)

                        delta_content = data.get("delta_content", "")
                        if delta_content:
                            content = _clean_thinking_content(delta_content)
                            thinking_chunk = create_openai_chunk(
                                chat_id,
                                model,
                                {
                                    "role": "assistant",
                                    "reasoning_content": content.replace("\n>", "\n"),
                                },
                            )
                            yield format_sse_chunk(thinking_chunk)

                    elif phase in ("answer", "other"):
                        _pre_delta_content = (
                            delta_content if delta_content else None
                        )
                        edit_content = data.get("edit_content", "")
                        delta_content = data.get("delta_content", "")

                        if edit_content:
                            with_detail = "</details>" in edit_content
                            if (
                                has_thinking
                                and phase == "answer"
                                and with_detail
                            ):
                                thinking_content_last = edit_content.split(
                                    _pre_delta_content
                                )[-1].replace("</details>", "")
                                sig_chunk = create_openai_chunk(
                                    chat_id,
                                    model,
                                    {
                                        "role": "assistant",
                                        "reasoning_content": thinking_content_last,
                                    },
                                )
                                yield format_sse_chunk(sig_chunk)
                            elif phase == "other":
                                if edit_content:
                                    content_chunk = create_openai_chunk(
                                        chat_id,
                                        model,
                                        {"role": "assistant", "content": edit_content},
                                    )
                                    yield format_sse_chunk(content_chunk)
                        elif delta_content:
                            if not has_thinking:
                                has_thinking = True
                                role_chunk = create_openai_chunk(
                                    chat_id, model, {"role": "assistant"}
                                )
                                yield format_sse_chunk(role_chunk)

                            content_chunk = create_openai_chunk(
                                chat_id,
                                model,
                                {"role": "assistant", "content": delta_content},
                            )
                            yield format_sse_chunk(content_chunk)

                        if data.get("usage"):
                            logger.info(
                                "完成响应 - 使用统计: {}", json.dumps(data["usage"])
                            )
                            finish_chunk = create_openai_chunk(
                                chat_id,
                                model,
                                {"role": "assistant", "content": ""},
                                "stop",
                            )
                            finish_chunk["usage"] = data["usage"]
                            yield format_sse_chunk(finish_chunk)

                    elif phase == "done":
                        yield "data: [DONE]\n\n"

                except json.JSONDecodeError as e:
                    logger.debug("JSON解析错误: {}, 内容: {}", e, chunk_str[:1000])
                except Exception as e:
                    logger.error(
                        "处理chunk错误: {}, chunk: {}", e, chunk_str[:1000]
                    )

        logger.info("SSE 流处理完成，共处理 {} 行数据", line_count)

    except Exception as e:
        logger.error("流式响应处理错误: {}", e)
        logger.error(traceback.format_exc())
        async for chunk in streaming_error_generator("流处理失败", "stream_error"):
            yield chunk
