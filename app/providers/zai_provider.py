#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 提供商适配器
"""

import json
import time
import uuid
import hmac
import hashlib
import base64
import httpx
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from app.providers.base import BaseProvider, ProviderConfig
from app.models.schemas import OpenAIRequest, Message
from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.user_agent import get_random_user_agent
from app.utils.sse_tool_handler import SSEToolHandler

logger = get_logger()


def get_zai_dynamic_headers(chat_id: str = "") -> Dict[str, str]:
    """
    生成 Z.AI 特定的动态浏览器 headers，包含随机 User-Agent
    使用通用的 UserAgent 工具，但添加 Z.AI 特定的业务逻辑

    Args:
        chat_id: 聊天 ID，用于生成正确的 Referer

    Returns:
        Dict[str, str]: 包含 Z.AI 特定配置的 headers
    """
    # 随机选择浏览器类型，偏向Chrome和Edge
    browser_choices = [
        "chrome",
        "chrome",
        "chrome",
        "edge",
        "edge",
        "firefox",
        "safari",
    ]
    browser_type = random.choice(browser_choices)

    user_agent = get_random_user_agent(browser_type)

    # 提取版本信息
    chrome_version = "139"
    edge_version = "139"

    if "Chrome/" in user_agent:
        try:
            chrome_version = user_agent.split("Chrome/")[1].split(".")[0]
            sec_ch_ua = f'"Google Chrome";v="{chrome_version}", "Not?A_Brand";v="8", "Chromium";v="{chrome_version}"'
        except:
            pass

    if "Edg/" in user_agent:
        try:
            edge_version = user_agent.split("Edg/")[1].split(".")[0]
            sec_ch_ua = f'"Microsoft Edge";v="{edge_version}", "Chromium";v="{chrome_version}", "Not_A Brand";v="24"'
        except:
            sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}"'
    elif "Firefox/" in user_agent:
        sec_ch_ua = None  # Firefox不使用sec-ch-ua
    else:
        sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}"'

    # Z.AI 特定的 headers
    headers = {
        # Core content negotiation
        "Content-Type": "application/json",
        "Accept": "*/*",
        # Connection/perf hints to reduce handshake overhead
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        # UA and app-specific headers
        "User-Agent": user_agent,
        "Accept-Language": "zh-CN",
        "DNT": "1",
        "Priority": "u=1, i",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-FE-Version": settings.X_FE_VERSION,
        "Origin": "https://chat.z.ai",
    }

    # 添加浏览器特定的 sec-ch-ua headers
    if sec_ch_ua:
        headers["sec-ch-ua"] = sec_ch_ua
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'

    # 根据 chat_id 设置 Referer
    if chat_id:
        headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        headers["Referer"] = "https://chat.z.ai/"

    return headers


class ZAIProvider(BaseProvider):
    """Z.AI 提供商"""

    def __init__(self):
        config = ProviderConfig(
            name="zai",
            api_endpoint=settings.API_ENDPOINT,
            timeout=30,
            headers=get_zai_dynamic_headers(),
        )
        super().__init__(config)

        # Z.AI 特定配置
        self.base_url = "https://chat.z.ai"
        self.auth_url = f"{self.base_url}/api/v1/auths/"

        # 存储客户端传递的 token
        self._client_token = None

        # 模型映射
        self.model_mapping = {
            settings.GLM45_MODEL: "0727-360B-API",  # GLM-4.5
            settings.GLM45_THINKING_MODEL: "0727-360B-API",  # GLM-4.5-Thinking
            settings.GLM45_SEARCH_MODEL: "0727-360B-API",  # GLM-4.5-Search
            settings.AIR_MODEL: "0727-106B-API",  # GLM-4.5-Air
            settings.GLM46_MODEL: "GLM-4-6-API-V1",  # GLM-4.6
            settings.GLM46_THINKING_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Thinking
            settings.GLM46_SEARCH_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Search
            settings.GLM47_MODEL: "glm-4.7",  # GLM-4.7
            settings.GLM47_THINKING_MODEL: "glm-4.7",  # GLM-4.7-Thinking
            settings.GLM47_SEARCH_MODEL: "glm-4.7",  # GLM-4.7-Search
            settings.GLM5_MODEL: "glm-5",  # GLM-5
            settings.GLM5_THINKING_MODEL: "glm-5",  # GLM-5-Thinking
            settings.GLM5_SEARCH_MODEL: "glm-5",  # GLM-5-Search
            settings.GLM5T_MODEL: "GLM-5-Turbo",  # GLM-5-Turbo
            settings.GLM5T_THINKING_MODEL: "GLM-5-Turbo",  # GLM-5-Turbo-Thinking
            settings.GLM5T_SEARCH_MODEL: "GLM-5-Turbo",  # GLM-5-Turbo-Search
            settings.GLM51_MODEL: "GLM-5.1",  # GLM-5.1
            settings.GLM51_THINKING_MODEL: "GLM-5.1",  # GLM-5.1-Thinking
            settings.GLM51_SEARCH_MODEL: "GLM-5.1",  # GLM-5.1-Search
        }

    def _generate_uuid(self) -> str:
        """生成UUID v4"""
        return str(uuid.uuid4())

    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            settings.GLM45_MODEL,
            settings.GLM45_THINKING_MODEL,
            settings.GLM45_SEARCH_MODEL,
            settings.AIR_MODEL,
            settings.GLM46_MODEL,
            settings.GLM46_THINKING_MODEL,
            settings.GLM46_SEARCH_MODEL,
            settings.GLM47_MODEL,
            settings.GLM47_THINKING_MODEL,
            settings.GLM47_SEARCH_MODEL,
            settings.GLM5_MODEL,
            settings.GLM5_THINKING_MODEL,
            settings.GLM5_SEARCH_MODEL,
            settings.GLM5T_MODEL,
            settings.GLM5T_THINKING_MODEL,
            settings.GLM5T_SEARCH_MODEL,
            settings.GLM51_MODEL,
            settings.GLM51_THINKING_MODEL,
            settings.GLM51_SEARCH_MODEL,
        ]

    async def get_token(self) -> str:
        """获取认证令牌"""
        # 如果启用客户端 token 模式且有客户端 token，优先使用
        if settings.USE_CLIENT_TOKEN and self._client_token:
            self.logger.debug(f"使用客户端传递的 token: {self._client_token[:20]}...")
            return self._client_token

        # 如果启用匿名模式，只尝试获取访客令牌
        if settings.ANONYMOUS_MODE:
            try:
                headers = get_zai_dynamic_headers()
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        self.auth_url, headers=headers, timeout=10.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        token = data.get("token", "")
                        if token:
                            self.logger.debug(f"获取访客令牌成功: {token[:20]}...")
                            return token
            except Exception as e:
                self.logger.warning(f"异步获取访客令牌失败: {e}")

            # 匿名模式下，如果获取访客令牌失败，直接返回空
            self.logger.error("❌ 匿名模式下获取访客令牌失败")
            return ""

        self.logger.error("❌ 无法获取有效的认证令牌")
        return ""

    def _generate_signature_params(
        self, token: str, user_message: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """为Z.AI请求生成签名和相关参数"""
        try:
            # 1. 解码JWT令牌
            parts = token.split(".")
            if len(parts) != 3:
                self.logger.error("无效的JWT令牌格式，无法生成签名")
                return None

            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            decoded_payload = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(decoded_payload)
            user_id = payload.get("id")
            if not user_id:
                self.logger.error("从JWT令牌中无法获取 'id'，无法生成签名")
                return None

            # 2. 生成签名所需参数
            timestamp = int(time.time() * 1000)
            request_id = self._generate_uuid()
            chat_id = self._generate_uuid()

            # 3. 构造签名
            # 签名1：时间及key
            time_5min_split = timestamp // (5 * 60 * 1000)
            key = settings.ZAI_SIGNATURE_KEY.encode("utf-8")
            signature_pre = hmac.new(
                key, str(time_5min_split).encode("utf-8"), hashlib.sha256
            ).hexdigest()
            # 签名2：对消息签名
            # 用户最后一条消息
            safe_user_message = user_message or ""
            # 带请求信息的拼接字段
            request_info = (
                f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
            )
            # 用户消息的Base64编码
            user_message_encode = base64.b64encode(
                safe_user_message.encode("utf-8")
            ).decode("utf-8")
            # 完整签名字符串
            sign_str = f"{request_info}|{user_message_encode}|{str(timestamp)}"
            signature = hmac.new(
                signature_pre.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            return {
                "user_id": user_id,
                "chat_id": chat_id,
                "request_id": request_id,
                "timestamp": timestamp,
                "signature": signature,
            }
        except Exception as e:
            self.logger.error(f"生成签名参数时出错: {e}")
            return None

    def _generate_browser_params(self, user_agent: str) -> Dict[str, Any]:
        """生成浏览器环境参数"""
        # 解析浏览器信息
        browser_name = "Chrome"
        os_name = "Windows"

        if "Chrome/" in user_agent:
            browser_name = "Chrome"
        elif "Edg/" in user_agent:
            browser_name = "Edge"
        elif "Firefox/" in user_agent:
            browser_name = "Firefox"
        elif "Safari/" in user_agent:
            browser_name = "Safari"

        if "Windows" in user_agent:
            os_name = "Windows"
        elif "Mac" in user_agent:
            os_name = "macOS"
        elif "Linux" in user_agent:
            os_name = "Linux"

        now = datetime.now()

        return {
            # 语言和时区（固定）
            "language": "zh-CN",
            "languages": "zh-CN,en-US",
            "timezone": "Asia/Shanghai",
            "timezone_offset": -480,
            # 时间（动态）
            "local_time": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "utc_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            # 浏览器环境（固定值，模拟常见环境）
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_width": "1920",
            "viewport_height": "869",
            "viewport_size": "1920x869",
            "color_depth": "32",
            "pixel_ratio": "1",
            # 设备检测（固定）
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "40",
            # 页面信息（固定）
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
            # 浏览器信息（从UA解析）
            "user_agent": user_agent,
            "browser_name": browser_name,
            "os_name": os_name,
        }

    async def _create_upstream_chat(
        self,
        prompt: str,
        model: str,
        token: str,
        headers: Dict[str, str],
        enable_thinking: bool,
        web_search: bool,
        user_message_id: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
    ) -> str:
        """为需要真实会话的模型创建上游 chat，并返回 chat_id。"""
        init_content = prompt or ""

        message_id = user_message_id or self._generate_uuid()
        timestamp_seconds = int(time.time())
        body = {
            "chat": {
                "id": "",
                "title": "新聊天",
                "models": [model],
                "params": {},
                "history": {
                    "messages": {
                        message_id: {
                            "id": message_id,
                            "parentId": None,
                            "childrenIds": [],
                            "role": "user",
                            "content": init_content,
                            "timestamp": timestamp_seconds,
                            "models": [model],
                        }
                    },
                    "currentId": message_id,
                },
                "tags": [],
                "flags": [],
                "features": [
                    {
                        "type": "tool_selector",
                        "server": "tool_selector_h",
                        "status": "hidden",
                    }
                ],
                "mcp_servers": list(mcp_servers or []),
                "enable_thinking": enable_thinking,
                "auto_web_search": web_search,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
            }
        }
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": headers.get("User-Agent", ""),
            "Accept-Language": headers.get("Accept-Language", "zh-CN"),
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/chats/new",
                headers=request_headers,
                json=body,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"上游创建 chat 失败: {response.status_code} {response.text}"
            )

        payload = response.json()
        chat_id = str(payload.get("id") or payload.get("chat", {}).get("id") or "")
        if not chat_id:
            raise RuntimeError("上游创建 chat 成功但未返回 chat_id")
        return chat_id

    async def _delete_upstream_chat(
        self,
        chat_id: str,
        token: str,
        headers: Dict[str, str],
    ) -> None:
        """删除已创建的上游 chat。"""
        if not chat_id or not token:
            self.logger.warning(
                f"跳过删除上游 chat，参数缺失: chat_id={chat_id or '空'}, token={'有' if token else '空'}"
            )
            return

        request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": headers.get("User-Agent", ""),
            "Accept-Language": headers.get(
                "Accept-Language", "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
            ),
            "Cache-Control": "no-cache",
            "DNT": headers.get("DNT", "1"),
            "Priority": headers.get("Priority", "u=1, i"),
            "sec-ch-ua": headers.get("sec-ch-ua", ""),
            "sec-ch-ua-mobile": headers.get("sec-ch-ua-mobile", "?0"),
            "sec-ch-ua-platform": headers.get("sec-ch-ua-platform", '"Windows"'),
            "Sec-Fetch-Dest": headers.get("Sec-Fetch-Dest", "empty"),
            "Sec-Fetch-Mode": headers.get("Sec-Fetch-Mode", "cors"),
            "Sec-Fetch-Site": headers.get("Sec-Fetch-Site", "same-origin"),
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/c/{chat_id}",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/v1/chats/{chat_id}",
                    headers=request_headers,
                )

            if response.is_success:
                self.logger.info(f"🗑️ 已删除上游 chat: {chat_id}")
            else:
                self.logger.warning(
                    f"⚠️ 删除上游 chat 失败: {response.status_code} {response.text}"
                )
        except Exception as e:
            self.logger.warning(f"⚠️ 删除上游 chat 异常: {e}")

    async def transform_request(self, request: OpenAIRequest) -> Dict[str, Any]:
        """转换OpenAI请求为Z.AI格式"""
        self.logger.info(f"🔄 转换 OpenAI 请求到 Z.AI 格式: {request.model}")
        original_messages = list(request.messages)
        requested_model = request.model
        requested_model_casefold = requested_model.casefold()
        requested_thinking_enable = (
            isinstance(request.thinking, dict)
            and request.thinking.get("type") == "enabled"
        )
        is_anthropic_messages = any(
            isinstance(msg.content, list) for msg in original_messages
        )
        is_search = "-search" in requested_model_casefold
        is_thinking = (
            is_anthropic_messages
            or requested_thinking_enable
            or ("-thinking" in requested_model_casefold)
        )

        def _extract_visible_text(content: Any) -> str:
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

        def _extract_reasoning_and_content(msg: Message) -> tuple[str, str]:
            content_text = _extract_visible_text(msg.content)
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
                        think_start + len("<think>") : think_end
                    ].strip()
                    content_text = content_text[think_end + len("</think>") :].lstrip()

            return reasoning_text, content_text

        def _render_tool_calls(tool_calls: Any) -> str:
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

        def _extract_tool_responses(content: Any) -> List[str]:
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

        def _get_user_message_content(messages: List[Message]) -> str:
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

        def _serialize_messages(messages: List[Message]) -> List[Dict[str, Any]]:
            serialized_messages = []
            for msg in messages:
                if isinstance(msg.content, str):
                    serialized_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )
                elif isinstance(msg.content, list):
                    content_parts = []
                    for part in msg.content:
                        if hasattr(part, "type") and hasattr(part, "text"):
                            content_parts.append({"type": part.type, "text": part.text})
                    serialized_messages.append(
                        {"role": msg.role, "content": content_parts}
                    )
            return serialized_messages

        # 0. 按 Z.AI 模板将 OpenAI messages 压平成单条 user 消息。
        if request.messages:
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
                    for tool_response in _extract_tool_responses(msg.content):
                        tool_message_parts.append(
                            f"<tool_response>{tool_response}</tool_response>"
                        )
                    merged_message_parts.append(
                        "\n".join(part for part in tool_message_parts if part).rstrip()
                    )
                    previous_role = role
                    continue

                reasoning_content, content_text = _extract_reasoning_and_content(msg)
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
                    tool_calls_text = _render_tool_calls(msg.tool_calls)
                    if tool_calls_text:
                        message_body_parts.append(tool_calls_text)

                message_body = "\n".join(part for part in message_body_parts if part)
                merged_message_parts.append(f"<|{role}|>\n{message_body}".rstrip())
                previous_role = role

            # 末尾补 assistant 前缀，显式告诉上游从 assistant 继续生成。
            generation_suffix = "<think>" if is_thinking else "</think>"
            merged_message_parts.append(f"<|assistant|>{generation_suffix}")

            merged_messages_content = "\n".join(
                part for part in merged_message_parts if part
            )
            request.messages = [Message(role="user", content=merged_messages_content)]

        # 1. 获取认证令牌
        token = await self.get_token()
        if not token:
            self.logger.error("❌ 无法获取认证令牌，请求将失败")
            return {
                "url": self.config.api_endpoint,
                "params": {},
                "headers": {},
                "body": {},
                "token": None,
                "chat_id": "no-chat-id",
                "model": request.model,
            }

        # 2. 提取签名和上游建会话所需的用户消息文本。
        user_message_content = _get_user_message_content(request.messages)
        if not user_message_content:
            self.logger.warning(
                "⚠️ 无法从请求中找到用户消息内容用于签名，将使用空字符串"
            )

        # 3. 生成签名及相关参数
        signature_params = self._generate_signature_params(token, user_message_content)
        if not signature_params:
            self.logger.error("❌ 生成签名失败，请求将失败")
            return {
                "url": self.config.api_endpoint,
                "params": {},
                "headers": {},
                "body": {},
                "token": token,
                "chat_id": "no-chat-id",
                "model": request.model,
            }

        user_id = signature_params["user_id"]
        chat_id = signature_params["chat_id"]
        request_id = signature_params["request_id"]
        timestamp = signature_params["timestamp"]
        signature = signature_params["signature"]

        # 4. 构建请求头
        headers = get_zai_dynamic_headers(chat_id)
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Signature"] = signature

        # 5. 构建请求 URL 和 Params
        url = f"{self.base_url}/api/v2/chat/completions"

        # 获取浏览器环境参数
        browser_params = self._generate_browser_params(headers.get("User-Agent", ""))

        params = {
            "timestamp": timestamp,
            "requestId": request_id,
            "user_id": user_id,
            "token": token,
            "current_url": f"{self.base_url}/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "signature_timestamp": timestamp,
            # 固定部分
            "version": "0.0.1",
            "platform": "web",
            # 添加浏览器环境参数
            **browser_params,
        }

        # 6. 将消息转换为上游请求体可直接使用的格式。
        messages = _serialize_messages(request.messages)

        # 7. 计算模型特性和上游模型 ID。
        if is_anthropic_messages:
            self.logger.info(f"content为List，开启思考： {is_anthropic_messages}")
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")

        # 8. 构建MCP服务器列表
        mcp_servers = []
        if is_search and "-4.5" in requested_model:
            mcp_servers.append("deep-web-search")
            self.logger.info("🔍 检测到搜索模型，添加 deep-web-search MCP 服务器")

        # 9. 先创建真实上游会话，再构造 completions 请求体。
        current_user_message_id = self._generate_uuid()
        self.logger.debug(
            f"请求模型：{requested_model}，上游模型ID：{upstream_model_id}，思考模式：{is_thinking}，搜索模式：{is_search}"
        )
        chat_id = await self._create_upstream_chat(
            prompt=user_message_content or "",
            model=upstream_model_id,
            token=token,
            headers=headers,
            enable_thinking=is_thinking,
            web_search=is_search,
            user_message_id=current_user_message_id,
            mcp_servers=mcp_servers,
        )
        headers["Referer"] = f"{self.base_url}/c/{chat_id}"
        params["current_url"] = f"{self.base_url}/c/{chat_id}"
        params["pathname"] = f"/c/{chat_id}"
        current_user_message_parent_id = None
        self.logger.info(
            f"已创建真实聊天会话，模型：{upstream_model_id}，ChatID：{chat_id}"
        )

        tools = (
            request.tools
            if settings.TOOL_SUPPORT and not is_thinking and request.tools
            else None
        )
        if tools:
            self.logger.info(f"启用工具支持: {len(tools)} 个工具")

        body = {
            "stream": True,  # 总是使用流式
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
            "id": self._generate_uuid(),
            "current_user_message_id": current_user_message_id,
            "current_user_message_parent_id": current_user_message_parent_id,
        }

        if tools:
            body["tools"] = tools
        else:
            body["tools"] = None

        if request.temperature is not None:
            body["params"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["params"]["max_tokens"] = request.max_tokens

        # 10. 返回转换后的请求对象。
        self.logger.debug(
            f"转换后的请求头:\n {json.dumps(headers, ensure_ascii=False, indent=2)}"
        )
        self.logger.debug(
            f"转换后的请求参数:\n {json.dumps(params, ensure_ascii=False, indent=2)}"
        )
        self.logger.debug(
            f"转换后的请求体:\n {json.dumps(body, ensure_ascii=False, indent=2)}"
        )
        return {
            "url": url,
            "params": params,
            "headers": headers,
            "body": body,
            "token": token,
            "chat_id": chat_id,
            "model": requested_model,
        }

    @staticmethod
    async def _streaming_error_generator(
        error_msg: str, error_type: str, code: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """为流式响应生成一个标准的错误块。"""
        error_payload: Dict[str, Any] = {"message": error_msg, "type": error_type}
        if code is not None:
            error_payload["code"] = code
        error_response = {"error": error_payload}
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

    async def chat_completion(
        self,
        request: OpenAIRequest,
        client_api_key: Optional[str] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """聊天完成接口"""
        # 存储客户端 token
        self._client_token = client_api_key
        self.log_request(request)

        try:
            # 1. 转换请求
            transformed = await self.transform_request(request)

            # 2. 检查转换是否成功（例如，是否获取到令牌）
            if not transformed.get("token"):
                error_msg = "请求转换失败，无法获取令牌"
                self.log_response(False, error_msg)
                if request.stream:
                    return self._streaming_error_generator(error_msg, "auth_error")
                else:
                    return self.handle_error(Exception(error_msg), "请求转换")

            # 3. 根据请求类型（流式/非流式）返回响应
            if request.stream:
                # 流式响应
                return self._create_stream_response(request, transformed)
            else:
                # 非流式响应
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        transformed["url"],
                        params=transformed.get("params", {}),
                        headers=transformed["headers"],
                        json=transformed["body"],
                    )
                try:
                    # transform_response 知道如何处理上游的流并聚合成单个响应
                    return await self.transform_response(response, request, transformed)
                finally:
                    if settings.AUTO_DELETE_UPSTREAM_CHAT:
                        await self._delete_upstream_chat(
                            transformed.get("chat_id", ""),
                            transformed.get("token", ""),
                            transformed.get("headers", {}),
                        )
                    else:
                        self.logger.warning(
                            f"保留上游 chat，未自动删除: {transformed.get('chat_id', '')}"
                        )

        except Exception as e:
            self.log_response(False, str(e))
            if request.stream:
                # 对于流式请求，在主函数中捕获异常时返回一个错误生成器
                return self._streaming_error_generator(
                    f"请求处理时发生未知错误: {e}", "internal_error"
                )
            else:
                return self.handle_error(e, "请求处理")

    async def _create_stream_response(
        self, request: OpenAIRequest, transformed: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式响应生成器"""
        current_token = transformed.get("token", "")
        cleanup_done = False

        async def _cleanup_upstream_chat() -> None:
            nonlocal cleanup_done
            if cleanup_done:
                return
            cleanup_done = True
            if not settings.AUTO_DELETE_UPSTREAM_CHAT:
                self.logger.warning(
                    f"保留上游 chat，未自动删除: {transformed.get('chat_id', '')}"
                )
                return
            await self._delete_upstream_chat(
                transformed.get("chat_id", ""),
                transformed.get("token", ""),
                transformed.get("headers", {}),
            )

        try:
            async with httpx.AsyncClient(
                timeout=60.0,
                http2=True,
            ) as client:
                self.logger.info(f"🎯 发送请求到 Z.AI: {transformed['url']}")
                async with client.stream(
                    "POST",
                    transformed["url"],
                    params=transformed.get("params", {}),
                    json=transformed["body"],
                    headers=transformed["headers"],
                ) as response:
                    # 检查响应状态码
                    if response.status_code != 200:
                        # 其他错误，直接返回
                        self.logger.error(f"❌ 上游返回错误: {response.status_code}")
                        error_text = await response.aread()
                        error_msg = error_text.decode("utf-8", errors="ignore")
                        if error_msg:
                            self.logger.error(f"❌ 错误详情: {error_msg}")
                        error_response = {
                            "error": {
                                "message": f"Upstream error: {response.status_code}",
                                "type": "upstream_error",
                                "code": response.status_code,
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        await _cleanup_upstream_chat()
                        return

                    # 处理流式响应
                    chat_id = transformed["chat_id"]
                    model = transformed["model"]
                    async for chunk in self._handle_stream_response(
                        response, chat_id, model, transformed
                    ):
                        yield chunk
                    return

        except Exception as e:
            self.logger.error(f"❌ 流处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            await _cleanup_upstream_chat()

            # 返回错误
            error_response = {"error": {"message": str(e), "type": "stream_error"}}
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
            return
        finally:
            await _cleanup_upstream_chat()

    async def transform_response(
        self,
        response: httpx.Response,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """转换Z.AI响应为OpenAI格式"""
        chat_id = transformed["chat_id"]
        model = transformed["model"]

        # 检查响应是否成功，如果不成功，提前处理错误
        if not response.is_success:
            error_msg = f"Z.AI API 错误: {response.status_code}"
            self.log_response(False, error_msg)
            # 对于非流式请求，直接返回错误字典
            if not request.stream:
                return self.handle_error(Exception(error_msg), "API响应")
            # 对于流式请求，返回一个错误生成器
            else:
                return self._streaming_error_generator(
                    error_msg, "upstream_error", response.status_code
                )

        if request.stream:
            return self._handle_stream_response(response, chat_id, model, transformed)
        else:
            return await self._handle_non_stream_response(response, chat_id, model)

    async def _handle_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        transformed: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """处理Z.AI流式响应"""
        self.logger.info(f"✅ Z.AI 响应成功，开始处理 SSE 流")

        has_tools = transformed["body"].get("tools") is not None
        tool_handler = None
        # Early ack: send an assistant role chunk immediately so the client sees progress
        try:
            role_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant"})
            yield await self.format_sse_chunk(role_chunk)
        except Exception:
            pass

        if has_tools:
            tool_handler = SSEToolHandler(model, stream=True)
            self.logger.info(
                f"🔧 初始化工具处理器: {len(transformed['body'].get('tools', []))} 个工具"
            )

        has_thinking = False
        delta_content = None
        buffer = ""
        line_count = 0
        self.logger.debug("📡 开始接收 SSE 流数据...")

        try:
            async for line in response.aiter_lines():
                line_count += 1
                if not line:
                    continue

                buffer += line + "\n"
                while "\n" in buffer:
                    current_line, buffer = buffer.split("\n", 1)
                    if not current_line.strip():
                        continue

                    if current_line.startswith("data:"):
                        chunk_str = current_line[5:].strip()
                        if not chunk_str or chunk_str == "[DONE]":
                            if chunk_str == "[DONE]" and not tool_handler:
                                yield "data: [DONE]\n\n"
                            continue

                        self.logger.debug(
                            f"📦 解析数据块: {chunk_str[:1000]}..."
                            if len(chunk_str) > 1000
                            else f"📦 解析数据块: {chunk_str}"
                        )

                        try:
                            chunk = json.loads(chunk_str)
                            if chunk.get("type") == "chat:completion":
                                data = chunk.get("data", {})
                                phase = data.get("phase")

                                if phase and phase != getattr(
                                    self, "_last_phase", None
                                ):
                                    self.logger.info(f"📈 SSE 阶段: {phase}")
                                    self._last_phase = phase

                                if tool_handler:
                                    sse_chunk = {
                                        "phase": phase,
                                        "edit_content": data.get("edit_content", ""),
                                        "delta_content": data.get("delta_content", ""),
                                        "edit_index": data.get("edit_index"),
                                        "usage": data.get("usage", {}),
                                    }
                                    for output in tool_handler.process_sse_chunk(
                                        sse_chunk
                                    ):
                                        yield output
                                elif phase == "thinking":
                                    if not has_thinking:
                                        has_thinking = True
                                        role_chunk = self.create_openai_chunk(
                                            chat_id, model, {"role": "assistant"}
                                        )
                                        yield await self.format_sse_chunk(role_chunk)

                                    delta_content = data.get("delta_content", "")
                                    if delta_content:
                                        # 处理思考内容格式
                                        content = (
                                            delta_content.split("</summary>\n>")[
                                                -1
                                            ].strip()
                                            if delta_content.startswith("<details")
                                            and "</summary>\n>" in delta_content
                                            else delta_content
                                        )

                                        thinking_chunk = self.create_openai_chunk(
                                            chat_id,
                                            model,
                                            {
                                                "role": "assistant",
                                                "reasoning_content": content.replace(
                                                    "\n>", "\n"
                                                ),
                                            },
                                        )
                                        yield await self.format_sse_chunk(
                                            thinking_chunk
                                        )
                                elif phase == "answer" or phase == "other":
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
                                            sig_chunk = self.create_openai_chunk(
                                                chat_id,
                                                model,
                                                {
                                                    "role": "assistant",
                                                    "reasoning_content": thinking_content_last,
                                                },
                                            )
                                            yield await self.format_sse_chunk(sig_chunk)
                                        elif phase == "other":
                                            content_after = edit_content
                                            if content_after:
                                                content_chunk = (
                                                    self.create_openai_chunk(
                                                        chat_id,
                                                        model,
                                                        {
                                                            "role": "assistant",
                                                            "content": content_after,
                                                        },
                                                    )
                                                )
                                                yield await self.format_sse_chunk(
                                                    content_chunk
                                                )
                                    elif delta_content:
                                        if not has_thinking:
                                            has_thinking = True  # Mark as true to prevent sending role chunk again
                                            role_chunk = self.create_openai_chunk(
                                                chat_id, model, {"role": "assistant"}
                                            )
                                            yield await self.format_sse_chunk(
                                                role_chunk
                                            )

                                        content_chunk = self.create_openai_chunk(
                                            chat_id,
                                            model,
                                            {
                                                "role": "assistant",
                                                "content": delta_content,
                                            },
                                        )
                                        yield await self.format_sse_chunk(content_chunk)

                                    if data.get("usage"):
                                        self.logger.info(
                                            f"📦 完成响应 - 使用统计: {json.dumps(data['usage'])}"
                                        )
                                        if not tool_handler:
                                            finish_chunk = self.create_openai_chunk(
                                                chat_id,
                                                model,
                                                {"role": "assistant", "content": ""},
                                                "stop",
                                            )
                                            finish_chunk["usage"] = data["usage"]
                                            yield await self.format_sse_chunk(
                                                finish_chunk
                                            )
                                elif phase == "done":
                                    yield "data: [DONE]\n\n"
                        except json.JSONDecodeError as e:
                            self.logger.debug(
                                f"❌ JSON解析错误: {e}, 内容: {chunk_str[:1000]}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"❌ 处理chunk错误: {e}, chunk: {chunk_str[:1000]}"
                            )

            if not tool_handler:
                self.logger.debug("📤 发送最终 [DONE] 信号 (如果尚未发送)")
                # The final DONE is now sent within the usage block to ensure it's the very last thing for non-tool responses.

            self.logger.info(f"✅ SSE 流处理完成，共处理 {line_count} 行数据")

        except Exception as e:
            self.logger.error(f"❌ 流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            async for chunk in self._streaming_error_generator(
                "流处理失败", "stream_error"
            ):
                yield chunk

    async def _handle_non_stream_response(
        self, response: httpx.Response, chat_id: str, model: str
    ) -> Dict[str, Any]:
        """处理非流式响应"""
        final_content = ""
        reasoning_content = ""
        usage_info: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            async for line in response.aiter_lines():
                if not line or not line.strip().startswith("data:"):
                    continue

                data_str = line.strip()[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    chunk = json.loads(data_str)
                    self.logger.debug(f"💡非流式内容：{chunk}")
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
                        cleaned = (
                            delta_content.split("</summary>\n>")[-1].strip()
                            if delta_content.startswith("<details")
                            and "</summary>\n>" in delta_content
                            else delta_content
                        )
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
            self.logger.error(f"❌ 非流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return self.handle_error(e, "非流式聚合")

        final_content = (final_content or "").strip()
        reasoning_content = (reasoning_content or "").strip()
        if not final_content and reasoning_content:
            final_content = reasoning_content

        return self.create_openai_response_with_reasoning(
            chat_id, model, final_content, reasoning_content, usage_info
        )
