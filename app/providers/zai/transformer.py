#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI → Z.AI 请求格式转换器

将 OpenAI API 格式的 messages 转换为 Z.AI 上游所需的格式。
"""

import json
from typing import Dict, List, Any

from app.models.schemas import Message


def extract_visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if (
                hasattr(part, "type")
                and part.type == "text"
                and hasattr(part, "text")
                and part.text
            ):
                text_parts.append(part.text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def extract_reasoning_and_content(msg: Message) -> tuple:
    content_text = extract_visible_text(msg.content)
    reasoning_text = (msg.reasoning_content or "").strip()

    if (
        not reasoning_text
        and "<think>" in content_text
        and "</think>" in content_text
    ):
        think_start = content_text.find("<think>")
        think_end = content_text.find("</think>", think_start)
        if think_end != -1:
            reasoning_text = content_text[
                think_start + len("<think>"):think_end
            ].strip()
            content_text = content_text[think_end + len("</think>"):].lstrip()

    return reasoning_text, content_text


def render_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""

    rendered_calls = []
    for tool_call in tool_calls:
        function_call = tool_call
        if isinstance(tool_call, dict) and tool_call.get("function"):
            function_call = tool_call["function"]

        if isinstance(function_call, dict):
            tool_name = function_call.get("name", "")
            arguments = function_call.get("arguments", {})
        else:
            tool_name = getattr(function_call, "name", "")
            arguments = getattr(function_call, "arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {"arguments": arguments}
        if not isinstance(arguments, dict):
            arguments = {"arguments": arguments}

        arg_parts = []
        for key, value in arguments.items():
            if isinstance(value, str):
                arg_value = value
            else:
                arg_value = json.dumps(value, ensure_ascii=False)
            arg_parts.append(
                f"<arg_key>{key}</arg_key><arg_value>{arg_value}</arg_value>"
            )

        rendered_calls.append(
            f"<tool_call>{tool_name}{''.join(arg_parts)}</tool_call>"
        )

    return "\n".join(rendered_calls)


def extract_tool_responses(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        responses = []
        for item in content:
            if hasattr(item, "output") and item.output is not None:
                responses.append(str(item.output))
            elif isinstance(item, dict) and item.get("output") is not None:
                responses.append(str(item["output"]))
            elif item is not None:
                responses.append(str(item))
        return responses
    if content is None:
        return []
    return [str(content)]


def get_user_message_text(messages: List[Message]) -> str:
    if not messages:
        return ""

    last_message = messages[-1]
    if isinstance(last_message.content, str):
        return last_message.content
    if isinstance(last_message.content, list):
        for part in reversed(last_message.content):
            if (
                hasattr(part, "type")
                and part.type == "text"
                and hasattr(part, "text")
            ):
                return part.text or ""
    return ""


def serialize_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    serialized = []
    for msg in messages:
        if isinstance(msg.content, str):
            serialized.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            content_parts = []
            for part in msg.content:
                if hasattr(part, "type") and hasattr(part, "text"):
                    content_parts.append({"type": part.type, "text": part.text})
            serialized.append({"role": msg.role, "content": content_parts})
    return serialized


def merge_messages_to_zai_format(
    messages: List[Message],
    is_thinking: bool,
) -> str:
    if not messages:
        return ""

    original_messages = list(messages)

    last_user_index = -1
    for index, msg in enumerate(original_messages):
        if (msg.role or "") == "user":
            last_user_index = index

    merged_message_parts = []
    previous_role = None
    for index, msg in enumerate(original_messages):
        role = msg.role or "user"
        if role == "tool":
            tool_message_parts = []
            if previous_role != "tool":
                tool_message_parts.append("<|observation|>")
            for tool_response in extract_tool_responses(msg.content):
                tool_message_parts.append(
                    f"<tool_response>{tool_response}</tool_response>"
                )
            merged_message_parts.append(
                "\n".join(part for part in tool_message_parts if part).rstrip()
            )
            previous_role = role
            continue

        reasoning_content, content_text = extract_reasoning_and_content(msg)
        message_body_parts = []

        if role == "assistant":
            keep_reasoning = reasoning_content and index > last_user_index
            if keep_reasoning:
                message_body_parts.append(
                    f"<think>{reasoning_content.strip()}</think>"
                )
            else:
                message_body_parts.append("</think>")

        if content_text.strip():
            message_body_parts.append(content_text.strip())

        if role == "assistant":
            tool_calls_text = render_tool_calls(msg.tool_calls)
            if tool_calls_text:
                message_body_parts.append(tool_calls_text)

        message_body = "\n".join(part for part in message_body_parts if part)
        merged_message_parts.append(f"<|{role}|>\n{message_body}".rstrip())
        previous_role = role

    generation_suffix = "<think>" if is_thinking else "</think>"
    merged_message_parts.append(f"<|assistant|>{generation_suffix}")

    return "\n".join(part for part in merged_message_parts if part)
