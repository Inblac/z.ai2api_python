#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 提供商编排层

组合 headers、signature、auth、chat_session、transformer、
sse_parser、sse_tool_handler、non_stream 各模块，对外提供统一入口。
"""

import json
import httpx
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from app.core.config import settings
from app.models.schemas import OpenAIRequest, Message
from app.utils.logger import get_logger

from app.providers.zai.headers import get_zai_dynamic_headers, generate_browser_params
from app.providers.zai.signature import generate_uuid, generate_signature_params
from app.providers.zai.auth import get_zai_token
from app.providers.zai.chat_session import create_upstream_chat, delete_upstream_chat, delete_all_upstream_chats
from app.providers.zai.transformer import (
    merge_messages_to_zai_format,
    serialize_messages,
    get_user_message_text,
    inject_tools_prompt,
)
from app.providers.zai._chunk_utils import (
    create_openai_chunk,
    format_sse_chunk,
    streaming_error_generator,
)
from app.providers.zai.sse_parser import parse_non_tool_sse_stream, parse_tool_prompt_sse_stream
from app.providers.zai.non_stream import aggregate_non_stream_response

logger = get_logger()

# 模型名 → 上游模型 ID
MODEL_MAPPING: Dict[str, str] = {
    "GLM-4.5": "0727-360B-API",
    "GLM-4.5-Thinking": "0727-360B-API",
    "GLM-4.5-Search": "0727-360B-API",
    "GLM-4.5-Air": "0727-106B-API",
    "GLM-4.6": "GLM-4-6-API-V1",
    "GLM-4.6-Thinking": "GLM-4-6-API-V1",
    "GLM-4.6-Search": "GLM-4-6-API-V1",
    "GLM-4.7": "glm-4.7",
    "GLM-4.7-Thinking": "glm-4.7",
    "GLM-4.7-Search": "glm-4.7",
    "GLM-5": "glm-5",
    "GLM-5-Thinking": "glm-5",
    "GLM-5-Search": "glm-5",
    "GLM-5-Turbo": "GLM-5-Turbo",
    "GLM-5-Turbo-Thinking": "GLM-5-Turbo",
    "GLM-5-Turbo-Search": "GLM-5-Turbo",
    "GLM-5.1": "GLM-5.1",
    "GLM-5.1-Thinking": "GLM-5.1",
    "GLM-5.1-Search": "GLM-5.1",
}

SUPPORTED_MODELS = list(MODEL_MAPPING.keys())

BASE_URL = "https://chat.z.ai"


class ZAIProvider:
    """Z.AI 提供商 — 编排层"""

    def __init__(self):
        self._client_token: Optional[str] = None

    def get_supported_models(self) -> List[str]:
        return SUPPORTED_MODELS

    async def chat_completion(
        self,
        request: OpenAIRequest,
        client_api_key: Optional[str] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        self._client_token = client_api_key
        logger.info(
            "处理请求: {}, 消息数: {}, 流式: {}",
            request.model, len(request.messages), request.stream,
        )

        try:
            transformed = await self._transform_request(request)

            if not transformed.get("token"):
                error_msg = "请求转换失败，无法获取令牌"
                logger.error(error_msg)
                if request.stream:
                    return streaming_error_generator(error_msg, "auth_error")
                else:
                    return {"error": {"message": error_msg, "type": "provider_error", "code": "internal_error"}}

            if request.stream:
                return self._create_stream_response(request, transformed)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        transformed["url"],
                        params=transformed.get("params", {}),
                        headers=transformed["headers"],
                        json=transformed["body"],
                    )
                if not response.is_success:
                    error_msg = f"Z.AI API 错误: {response.status_code}"
                    logger.error(error_msg)
                    return {"error": {"message": error_msg, "type": "upstream_error", "code": response.status_code}}
                body_lines = response.text.splitlines()
                logger.debug("上游非流式响应: status={}, body={}", response.status_code, body_lines[:50])
                try:
                    async def _line_gen():
                        for line in body_lines:
                            yield line
                    return await aggregate_non_stream_response(
                        _line_gen(),
                        transformed["chat_id"],
                        transformed["model"],
                        has_tools=transformed.get("has_tools", False),
                    )
                finally:
                    if settings.AUTO_DELETE_UPSTREAM_CHAT:
                        await delete_all_upstream_chats(
                            transformed.get("token", ""),
                            transformed.get("headers", {}),
                        )

        except Exception as e:
            logger.error("请求处理失败: {}", e)
            if request.stream:
                return streaming_error_generator(
                    f"请求处理时发生未知错误: {e}", "internal_error"
                )
            else:
                return {"error": {"message": f"请求处理 错误: {str(e)}", "type": "provider_error", "code": "internal_error"}}

    async def _transform_request(self, request: OpenAIRequest) -> Dict[str, Any]:
        logger.info("转换 OpenAI 请求到 Z.AI 格式: {}", request.model)
        original_messages = list(request.messages)
        requested_model = request.model

        requested_thinking_enable = (
            isinstance(request.thinking, dict)
            and request.thinking.get("type") == "enabled"
        )
        is_anthropic_messages = any(
            isinstance(msg.content, list) for msg in original_messages
        )
        is_search = "-search" in requested_model.casefold()
        is_thinking = (
            is_anthropic_messages
            or requested_thinking_enable
            or ("-thinking" in requested_model.casefold())
        )

        has_tools = bool(
            settings.TOOL_SUPPORT and not is_thinking and request.tools
        )

        if has_tools:
            original_messages = inject_tools_prompt(original_messages, request.tools)
            logger.info("注入工具提示词: {} 个工具", len(request.tools))

        if request.messages:
            merged_messages_content = merge_messages_to_zai_format(
                original_messages, is_thinking
            )
            request.messages = [
                Message(role="user", content=merged_messages_content)
            ]

        token = await get_zai_token(self._client_token)
        if not token:
            logger.error("无法获取认证令牌，请求将失败")
            return {"url": "", "params": {}, "headers": {}, "body": {}, "token": None, "chat_id": "no-chat-id", "model": request.model}

        user_message_content = get_user_message_text(request.messages)
        signature_params = generate_signature_params(token, user_message_content)
        if not signature_params:
            logger.error("生成签名失败，请求将失败")
            return {"url": "", "params": {}, "headers": {}, "body": {}, "token": token, "chat_id": "no-chat-id", "model": request.model}

        user_id = signature_params["user_id"]
        chat_id = signature_params["chat_id"]
        request_id = signature_params["request_id"]
        timestamp = signature_params["timestamp"]
        signature = signature_params["signature"]

        headers = get_zai_dynamic_headers(chat_id)
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Signature"] = signature

        url = f"{BASE_URL}/api/v2/chat/completions"
        browser_params = generate_browser_params(headers.get("User-Agent", ""))

        params: Dict[str, Any] = {
            "timestamp": timestamp,
            "requestId": request_id,
            "user_id": user_id,
            "token": token,
            "current_url": f"{BASE_URL}/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "signature_timestamp": timestamp,
            "version": "0.0.1",
            "platform": "web",
            **browser_params,
        }

        messages = serialize_messages(request.messages)

        upstream_model_id = MODEL_MAPPING.get(requested_model, "0727-360B-API")

        mcp_servers: List[str] = []
        if is_search and "-4.5" in requested_model:
            mcp_servers.append("deep-web-search")
            logger.info("检测到搜索模型，添加 deep-web-search MCP 服务器")

        current_user_message_id = generate_uuid()
        logger.debug(
            "请求模型：{}，上游模型ID：{}，思考模式：{}，搜索模式：{}",
            requested_model, upstream_model_id, is_thinking, is_search,
        )
        chat_id = await create_upstream_chat(
            prompt=user_message_content or "",
            upstream_model_id=upstream_model_id,
            token=token,
            user_agent=headers.get("User-Agent", ""),
            enable_thinking=is_thinking,
            web_search=is_search,
            user_message_id=current_user_message_id,
            mcp_servers=mcp_servers,
        )
        headers["Referer"] = f"{BASE_URL}/c/{chat_id}"
        params["current_url"] = f"{BASE_URL}/c/{chat_id}"
        params["pathname"] = f"/c/{chat_id}"

        body: Dict[str, Any] = {
            "stream": request.stream,
            "model": upstream_model_id,
            "messages": messages,
            "signature_prompt": user_message_content,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": is_search,
                "auto_web_search": is_search,
                "preview_mode": True,
                "flags": [],
                "vlm_tools_enable": False,
                "vlm_web_search_enable": False,
                "vlm_website_mode": False,
                "enable_thinking": is_thinking,
            },
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
            },
            "mcp_servers": mcp_servers,
            "variables": {
                "{{USER_NAME}}": "ProUltra",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}": datetime.now().strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}": datetime.now().strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}": datetime.now().strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
                "{{USER_LANGUAGE}}": "zh-CN",
            },
            "chat_id": chat_id,
            "id": generate_uuid(),
            "current_user_message_id": current_user_message_id,
            "current_user_message_parent_id": None,
        }

        if request.temperature is not None:
            body["params"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["params"]["max_tokens"] = request.max_tokens

        logger.debug(
            "转换后的请求头:\n{}",
            json.dumps(headers, ensure_ascii=False, indent=2),
        )
        logger.debug(
            "转换后的请求参数:\n{}",
            json.dumps(params, ensure_ascii=False, indent=2),
        )
        logger.debug(
            "转换后的请求体:\n{}",
            json.dumps(body, ensure_ascii=False, indent=2),
        )

        return {
            "url": url,
            "params": params,
            "headers": headers,
            "body": body,
            "token": token,
            "chat_id": chat_id,
            "model": requested_model,
            "has_tools": has_tools,
        }

    async def _create_stream_response(
        self, request: OpenAIRequest, transformed: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        cleanup_done = False

        async def _cleanup():
            nonlocal cleanup_done
            if cleanup_done:
                return
            cleanup_done = True
            if not settings.AUTO_DELETE_UPSTREAM_CHAT:
                return
            await delete_all_upstream_chats(
                transformed.get("token", ""),
                transformed.get("headers", {}),
            )

        try:
            async with httpx.AsyncClient(timeout=60.0, http2=True) as client:
                logger.info("发送请求到 Z.AI: {}", transformed["url"])
                async with client.stream(
                    "POST",
                    transformed["url"],
                    params=transformed.get("params", {}),
                    json=transformed["body"],
                    headers=transformed["headers"],
                ) as response:
                    if response.status_code != 200:
                        logger.error("上游返回错误: {}", response.status_code)
                        error_text = await response.aread()
                        error_msg = error_text.decode("utf-8", errors="ignore")
                        if error_msg:
                            logger.error("错误详情: {}", error_msg)
                        error_response = {
                            "error": {
                                "message": f"Upstream error: {response.status_code}",
                                "type": "upstream_error",
                                "code": response.status_code,
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        await _cleanup()
                        return

                    chat_id = transformed["chat_id"]
                    model = transformed["model"]
                    has_tools = transformed.get("has_tools", False)

                    if has_tools:
                        logger.info("初始化工具提示词注入模式解析器")
                        try:
                            role_chunk = create_openai_chunk(
                                chat_id, model, {"role": "assistant"}
                            )
                            yield format_sse_chunk(role_chunk)
                        except Exception:
                            pass
                        async for chunk in parse_tool_prompt_sse_stream(
                            response.aiter_lines(), chat_id, model
                        ):
                            yield chunk
                    else:
                        # 非工具模式：委托给 sse_parser
                        try:
                            role_chunk = create_openai_chunk(
                                chat_id, model, {"role": "assistant"}
                            )
                            yield format_sse_chunk(role_chunk)
                        except Exception:
                            pass
                        async for chunk in parse_non_tool_sse_stream(
                            response.aiter_lines(), chat_id, model
                        ):
                            yield chunk
                    return

        except Exception as e:
            logger.error("流处理错误: {}", e)
            logger.error(traceback.format_exc())
            await _cleanup()
            error_response = {"error": {"message": str(e), "type": "stream_error"}}
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
            return
        finally:
            await _cleanup()



