#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI → Z.AI 请求格式转换器

将 OpenAI API 格式的 messages 转换为 Z.AI 上游所需的格式。
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional

from app.models.schemas import Message


def inject_tools_prompt(messages: List[Message], tools: List[Dict[str, Any]]) -> List[Message]:
    """
    将 OpenAI tools 定义注入为系统提示词，避免 body 中携带 tools 导致上游拒绝。
    """
    if not tools:
        return list(messages)

    tools_desc_parts = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        description = fn.get("description", "")
        parameters = fn.get("parameters", {})
        tools_desc_parts.append(
            f"### {name}\n"
            f"Description: {description}\n"
            f"Parameters: {json.dumps(parameters, ensure_ascii=False, indent=2)}"
        )
    tools_desc = "\n\n".join(tools_desc_parts)

    prompt = (
        "You are an assistant with access to tools. When you need to use a tool, you MUST output ONLY a single JSON object with NO markdown, NO explanations, and NO extra text.\n"
        "\n"
        "STRICT RULES:\n"
        "1. If a tool is needed, output EXACTLY this format (nothing else):\n"
        '{"tool_calls":[{"name":"TOOL_NAME","arguments":{"param":"value"}}]}\n'
        "\n"
        "2. Do NOT wrap the JSON in markdown code blocks (no ```json).\n"
        "3. Do NOT add any explanation before or after the JSON.\n"
        "4. If no tool is needed, respond normally with plain text.\n"
        "\n"
        "Available tools:\n"
        f"{tools_desc}\n"
        "\n"
        "Examples:\n"
        "User: What is the weather in Beijing?\n"
        'Assistant: {"tool_calls":[{"name":"get_weather","arguments":{"location":"Beijing"}}]}\n'
        "\n"
        "User: Hello\n"
        "Assistant: Hello! How can I help you today?"
    )

    new_messages = list(messages)
    for i, msg in enumerate(new_messages):
        if msg.role == "system":
            original = extract_visible_text(msg.content)
            new_messages[i] = Message(role="system", content=original + "\n\n" + prompt)
            return new_messages

    new_messages.insert(0, Message(role="system", content=prompt))
    return new_messages


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


def extract_json_object(text: str, key: str) -> str | None:
    """从字符串中提取包含指定 key 的完整 JSON 对象，支持嵌套括号匹配。"""
    idx = text.find(f'"{key}"')
    if idx == -1:
        return None
    start = idx
    while start > 0 and text[start] != "{":
        start -= 1
    if text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def normalize_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    tool_calls = []
    for idx, tc in enumerate(raw_tool_calls or []):
        if not isinstance(tc, dict):
            continue

        fn = tc.get("function", {})
        name = tc.get("name") or fn.get("name", "")
        args = tc.get("arguments")
        if args is None and isinstance(fn, dict):
            args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}_{idx}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        })
    return tool_calls


def parse_tool_calls(content: str) -> Dict[str, Any]:
    """
    从模型输出文本中解析工具调用。

    三层回退策略：
    1. 去除 Markdown 代码块 → 精确 JSON 提取
    2. 修复常见 JSON 格式错误（单引号→双引号、未加引号的 key）
    返回 {"tool_calls": list | None, "text": str}
    """
    if not content or not content.strip():
        return {"tool_calls": None, "text": content}

    working = content.strip()
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', working)
    if code_block_match:
        working = code_block_match.group(1).strip()

    brace_match = extract_json_object(working, "tool_calls")
    if brace_match:
        try:
            parsed = json.loads(brace_match)
            if (
                isinstance(parsed.get("tool_calls"), list)
                and len(parsed["tool_calls"]) > 0
            ):
                tool_calls = normalize_tool_calls(parsed["tool_calls"])
                text = content.replace(brace_match, "").strip()
                if code_block_match:
                    text = content.replace(code_block_match.group(0), "").strip()
                return {"tool_calls": tool_calls, "text": text}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    try:
        fixed = working
        fixed = re.sub(r"(['\"])?([a-zA-Z0-9_]+)(['\"])?\s*:", r'"\2":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r':"\1"', fixed)
        parsed = json.loads(fixed)
        if (
            isinstance(parsed.get("tool_calls"), list)
            and len(parsed["tool_calls"]) > 0
        ):
            tool_calls = normalize_tool_calls(parsed["tool_calls"])
            text = content.replace(working, "").strip()
            if code_block_match:
                text = content.replace(code_block_match.group(0), "").strip()
            return {"tool_calls": tool_calls, "text": text}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return {"tool_calls": None, "text": content}
