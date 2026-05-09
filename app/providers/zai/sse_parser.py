#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI SSE 流解析器

处理 Z.AI SSE 流中的 tool_call / tool_response / thinking / answer / other / done 阶段，
生成 OpenAI 兼容的流式 chunk。
"""

import json
import re
import traceback
from typing import AsyncGenerator

from app.providers.zai._chunk_utils import (
    create_openai_chunk,
    format_sse_chunk,
    streaming_error_generator,
)
from app.providers.zai.transformer import parse_tool_calls
from app.utils.logger import get_logger

logger = get_logger()

_PUA_CHARS_RE = re.compile(r"[\ue000-\uf8ff]")
_CITATION_RE = re.compile(r"【turn0search\d+】")
_PARTIAL_CITATION_RE = re.compile(r"【[^】]*$")


def _strip_citations(text: str, pending: list[str]) -> tuple[str, list[str]]:
    """剔除引用标记，支持跨 chunk 的截断标记。

    将 pending 中积累的截断标记与当前文本拼接，剔除完整的
    ``【turn0search数字】`` 引用，最后截掉行尾未闭合的 ``【`` 存入 pending。
    返回 (清理后文本, 新的 pending 列表)。
    """
    combined = "".join(pending) + text
    combined = _CITATION_RE.sub("", combined)
    m = _PARTIAL_CITATION_RE.search(combined)
    if m:
        return combined[: m.start()], [m.group()]
    return combined, []


def _clean_thinking_content(delta_content: str) -> str:
    """清洗 thinking 阶段的 delta_content，去除 <details> 包裹层。"""
    if delta_content.startswith("<details") and "</summary>\n>" in delta_content:
        return delta_content.split("</summary>\n>")[-1].strip()
    return delta_content


def _format_search_results(metadata: dict | None) -> str | None:
    """从 metadata 中提取搜索结果，格式化为 Markdown。

    解析 ``metadata.browser.search_result`` 数组，取 title/url/text 三个字段，
    按 ``## 搜索结果`` / ``### [标题](url)`` / ``> 摘要`` 结构组织。
    若无搜索结果返回 None。
    """
    if not metadata:
        return None
    browser = metadata.get("browser") or {}
    results = browser.get("search_result", [])
    if not results:
        return None

    parts = ["## 搜索结果\n"]
    for item in results:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        text = item.get("text", "").strip()
        if url:
            parts.append(f"### [{title}]({url})")
        else:
            parts.append(f"### {title}")
        if text:
            text = _PUA_CHARS_RE.sub("", text)
            parts.append(f"> {text}\n")
    return "\n".join(parts)


def _tool_response_sse(data: dict, chat_id: str, model: str) -> str | None:
    """将 tool_response 阶段的搜索结果转为 SSE chunk。

    从 data.metadata 中提取搜索结果并格式化为 reasoning_content 的 SSE 行。
    无有效结果时返回 None。
    """
    formatted = _format_search_results(data.get("metadata"))
    if not formatted:
        return None
    return format_sse_chunk(create_openai_chunk(chat_id, model, {"role": "assistant", "reasoning_content": formatted}))


def _thinking_sse(data: dict, chat_id: str, model: str) -> str | None:
    """将 thinking 阶段的 delta_content 转为 reasoning_content SSE chunk。

    清洗 <details> 包裹层后发出，无内容时返回 None。
    """
    dc = data.get("delta_content", "")
    if not dc:
        return None
    content = _clean_thinking_content(dc).replace("\n>", "\n")
    return format_sse_chunk(create_openai_chunk(chat_id, model, {"role": "assistant", "reasoning_content": content}))


def _role_sse(chat_id: str, model: str) -> str:
    """生成空的 assistant role SSE chunk，标记流式输出的起始角色。"""
    return format_sse_chunk(create_openai_chunk(chat_id, model, {"role": "assistant"}))


def _finish_sse(chat_id: str, model: str, usage: dict) -> str:
    """生成结束 SSE chunk，携带 finish_reason="stop" 和可选的 usage 统计。"""
    finish = create_openai_chunk(chat_id, model, {"role": "assistant", "content": ""}, "stop")
    if usage:
        finish["usage"] = usage
    return format_sse_chunk(finish)


def _content_sse(text: str, chat_id: str, model: str) -> str:
    """生成普通文本 content SSE chunk。"""
    return format_sse_chunk(create_openai_chunk(chat_id, model, {"role": "assistant", "content": text}))


def _thinking_with_edit_sse(edit_content: str, pre_delta: str | None, chat_id: str, model: str) -> str:
    """从 edit_content 中提取 thinking 尾部并生成 SSE chunk。

    用 pre_delta 切分后取最后一段，去除 ``</details>`` 标记，
    以 reasoning_content 发出。
    """
    thinking = edit_content.split(pre_delta)[-1].replace("</details>", "")
    return format_sse_chunk(create_openai_chunk(chat_id, model, {"role": "assistant", "reasoning_content": thinking}))


async def parse_non_tool_sse_stream(
    lines: AsyncGenerator[str, None],
    chat_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """处理非工具模式的 SSE 流，生成 OpenAI 兼容 chunk。

    按阶段处理：tool_call 忽略，tool_response 输出搜索结果，
    thinking 输出推理内容，answer/other 输出正文并剔除引用标记，
    done 输出结束标记和 usage 统计。
    """
    logger.info("开始处理 SSE 流（非工具模式）")
    has_thinking = False
    delta_content = None
    buffer = ""
    line_count = 0
    last_phase = None
    usage_data = {}
    citation_pending: list[str] = []

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

                try:
                    chunk = json.loads(chunk_str)
                    if chunk.get("type") != "chat:completion":
                        continue

                    data = chunk.get("data", {})
                    phase = data.get("phase")

                    if phase and phase != last_phase:
                        logger.info("SSE 阶段: {}", phase)
                        last_phase = phase

                    logger.debug(
                        "解析数据块: {}",
                        chunk_str[:1000] + "..." if len(chunk_str) > 1000 else chunk_str,
                    )
                    
                    if data.get("usage"):
                        usage_data = data["usage"]

                    if phase == "tool_call":
                        continue

                    elif phase == "tool_response":
                        if not has_thinking:
                            has_thinking = True
                            yield _role_sse(chat_id, model)
                        sse = _tool_response_sse(data, chat_id, model)
                        if sse:
                            yield sse

                    elif phase == "thinking":
                        if not has_thinking:
                            has_thinking = True
                            yield _role_sse(chat_id, model)
                        sse = _thinking_sse(data, chat_id, model)
                        if sse:
                            yield sse

                    elif phase in ("answer", "other"):
                        _pre_delta_content = delta_content if delta_content else None
                        edit_content = data.get("edit_content", "")
                        delta_content = data.get("delta_content", "")

                        if edit_content:
                            with_detail = "</details>" in edit_content
                            if has_thinking and phase == "answer" and with_detail:
                                sse = _thinking_with_edit_sse(edit_content, _pre_delta_content, chat_id, model)
                                yield sse
                            elif phase == "other":
                                if edit_content:
                                    cleaned, citation_pending = _strip_citations(edit_content, citation_pending)
                                    if cleaned:
                                        yield _content_sse(cleaned, chat_id, model)
                        elif delta_content:
                            if not has_thinking:
                                has_thinking = True
                                yield _role_sse(chat_id, model)
                            cleaned, citation_pending = _strip_citations(delta_content, citation_pending)
                            if cleaned:
                                yield _content_sse(cleaned, chat_id, model)

                    elif phase == "done":
                        if usage_data:
                            logger.info("完成响应 - 使用统计: {}", json.dumps(usage_data))
                        yield _finish_sse(chat_id, model, usage_data)
                        yield "data: [DONE]\n\n"

                except json.JSONDecodeError as e:
                    logger.debug("JSON解析错误: {}, 内容: {}", e, chunk_str[:1000])
                except Exception as e:
                    logger.error("处理chunk错误: {}, chunk: {}", e, chunk_str[:1000])

        logger.info("SSE 流处理完成，共处理 {} 行数据", line_count)

    except Exception as e:
        logger.error("流式响应处理错误: {}", e)
        logger.error(traceback.format_exc())
        async for chunk in streaming_error_generator("流处理失败", "stream_error"):
            yield chunk


async def parse_tool_prompt_sse_stream(
    lines: AsyncGenerator[str, None],
    chat_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    处理启用了工具调用的 SSE 流（提示词注入模式）。

    工具调用以纯文本 JSON 出现在 answer 阶段。
    使用智能缓冲策略：普通文本直接透传，疑似 JSON 的文本缓冲后延迟判断。
    """
    logger.info("开始处理 SSE 流（工具提示词注入模式）")
    full_content = ""
    might_be_tool_call = False
    is_tool_call_mode = False
    pending_content = ""
    has_thinking = False
    delta_content = None
    last_phase = None
    usage_data = {}
    line_count = 0
    citation_pending: list[str] = []

    try:
        async for line in lines:
            line_count += 1
            if not line:
                continue

            if not line.startswith("data:"):
                continue

            chunk_str = line[5:].strip()
            if not chunk_str or chunk_str == "[DONE]":
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

                if data.get("usage"):
                    usage_data = data["usage"]

                if phase == "tool_call":
                    continue

                elif phase == "tool_response":
                    if not has_thinking:
                        has_thinking = True
                        yield _role_sse(chat_id, model)
                    sse = _tool_response_sse(data, chat_id, model)
                    if sse:
                        yield sse

                elif phase == "thinking":
                    if not has_thinking:
                        has_thinking = True
                        yield _role_sse(chat_id, model)
                    sse = _thinking_sse(data, chat_id, model)
                    if sse:
                        yield sse

                elif phase in ("answer", "other"):
                    _pre_delta_content = delta_content if delta_content else None
                    edit_content = data.get("edit_content", "")
                    delta_content = data.get("delta_content", "")

                    if edit_content:
                        with_detail = "</details>" in edit_content
                        if has_thinking and phase == "answer" and with_detail:
                            sse = _thinking_with_edit_sse(edit_content, _pre_delta_content, chat_id, model)
                            yield sse
                        elif phase == "other":
                            if edit_content:
                                full_content += edit_content
                                if is_tool_call_mode:
                                    continue
                                trimmed = full_content.strip()
                                if not might_be_tool_call:
                                    if trimmed.startswith("{"):
                                        might_be_tool_call = True
                                        pending_content += edit_content
                                    else:
                                        cleaned, citation_pending = _strip_citations(edit_content, citation_pending)
                                        if cleaned:
                                            yield _content_sse(cleaned, chat_id, model)
                                else:
                                    pending_content += edit_content
                                    if len(trimmed) >= 20:
                                        if (
                                            '"tool_calls"' in trimmed
                                            or "'tool_calls'" in trimmed
                                            or "tool_calls" in trimmed
                                        ):
                                            is_tool_call_mode = True
                                            pending_content = ""
                                        else:
                                            might_be_tool_call = False
                                            cleaned, citation_pending = _strip_citations(
                                                pending_content, citation_pending
                                            )
                                            if cleaned:
                                                yield _content_sse(cleaned, chat_id, model)
                                            pending_content = ""
                    elif delta_content:
                        full_content += delta_content
                        if is_tool_call_mode:
                            continue

                        if not has_thinking:
                            has_thinking = True
                            yield _role_sse(chat_id, model)

                        trimmed = full_content.strip()
                        if not might_be_tool_call:
                            if trimmed.startswith("{"):
                                might_be_tool_call = True
                                pending_content += delta_content
                            else:
                                cleaned, citation_pending = _strip_citations(delta_content, citation_pending)
                                if cleaned:
                                    yield _content_sse(cleaned, chat_id, model)
                        else:
                            pending_content += delta_content
                            if len(trimmed) >= 20:
                                if '"tool_calls"' in trimmed or "'tool_calls'" in trimmed or "tool_calls" in trimmed:
                                    is_tool_call_mode = True
                                    pending_content = ""
                                else:
                                    might_be_tool_call = False
                                    cleaned, citation_pending = _strip_citations(pending_content, citation_pending)
                                    if cleaned:
                                        yield _content_sse(cleaned, chat_id, model)
                                    pending_content = ""

                elif phase == "done":
                    if is_tool_call_mode:
                        parsed = parse_tool_calls(full_content)
                        if parsed.get("tool_calls"):
                            tool_calls_chunk = create_openai_chunk(
                                chat_id,
                                model,
                                {"role": "assistant", "tool_calls": parsed["tool_calls"]},
                                finish_reason="tool_calls",
                            )
                            if usage_data:
                                tool_calls_chunk["usage"] = usage_data
                            yield format_sse_chunk(tool_calls_chunk)
                    else:
                        yield _finish_sse(chat_id, model, usage_data)
                    yield "data: [DONE]\n\n"

            except json.JSONDecodeError as e:
                logger.debug("JSON解析错误: {}, 内容: {}", e, chunk_str[:1000])
            except Exception as e:
                logger.error("处理chunk错误: {}, chunk: {}", e, chunk_str[:1000])

        logger.info("SSE 流处理完成，共处理 {} 行数据", line_count)

    except Exception as e:
        logger.error("流式响应处理错误: {}", e)
        logger.error(traceback.format_exc())
        async for chunk in streaming_error_generator("流处理失败", "stream_error"):
            yield chunk
