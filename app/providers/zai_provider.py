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
from app.models.schemas import OpenAIRequest
from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.token_pool import get_token_pool
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
    browser_choices = ["chrome", "chrome", "chrome", "edge", "edge", "firefox", "safari"]
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
        # UA and app-specific headers
        "User-Agent": user_agent,
        "Accept-Language": "zh-CN",
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
            name="zai", api_endpoint=settings.API_ENDPOINT, timeout=30, headers=get_zai_dynamic_headers()
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
            settings.GLM46_GLM45_THINKING_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Thinking
            settings.GLM46_GLM45_SEARCH_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Search
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
            settings.GLM46_GLM45_THINKING_MODEL,
            settings.GLM46_GLM45_SEARCH_MODEL,
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
                    response = await client.get(self.auth_url, headers=headers, timeout=10.0)
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

        # 非匿名模式：首先使用token池获取备份令牌
        token_pool = get_token_pool()
        if token_pool:
            token = token_pool.get_next_token()
            if token:
                self.logger.debug(f"从token池获取令牌: {token[:20]}...")
                return token

        self.logger.error("❌ 无法获取有效的认证令牌")
        return ""

    def mark_token_failure(self, token: str, error: Optional[Exception] = None):
        """标记token使用失败"""
        token_pool = get_token_pool()
        if token_pool:
            error_to_report = (
                error if error is not None else Exception("Token failure reported without specific error")
            )
            token_pool.mark_token_failure(token, error_to_report)

    def _generate_signature_params(self, token: str, user_message: Optional[str]) -> Optional[Dict[str, Any]]:
        """为Z.AI请求生成签名和相关参数"""
        try:
            # 1. 解码JWT令牌
            parts = token.split('.')
            if len(parts) != 3:
                self.logger.error("无效的JWT令牌格式，无法生成签名")
                return None

            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding

            decoded_payload = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(decoded_payload)
            user_id = payload.get('id')
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
            key = settings.ZAI_SIGNATURE_KEY.encode('utf-8')
            signature_pre = hmac.new(key, str(time_5min_split).encode('utf-8'), hashlib.sha256).hexdigest()
            # 签名2：对消息签名
            # 用户最后一条消息
            safe_user_message = user_message or ""
            # 带请求信息的拼接字段
            request_info = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
            # 用户消息的Base64编码
            user_message_encode = base64.b64encode(safe_user_message.encode('utf-8')).decode('utf-8')
            # 完整签名字符串
            sign_str = f"{request_info}|{user_message_encode}|{str(timestamp)}"
            signature = hmac.new(signature_pre.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).hexdigest()

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
            # "timezone": "Asia/Shanghai",
            # "timezone_offset": -480,
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
            "title": "Z.ai Chat - Free AI powered by GLM-4.6 & GLM-4.5",
            # 浏览器信息（从UA解析）
            "user-agent": user_agent,
            "browser_name": browser_name,
            "os_name": os_name,
        }

    async def transform_request(self, request: OpenAIRequest) -> Dict[str, Any]:
        """转换OpenAI请求为Z.AI格式"""
        self.logger.info(f"🔄 转换 OpenAI 请求到 Z.AI 格式: {request.model}")

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
        self._current_token = token

        # 2. 获取用于签名的用户消息
        user_message_content = ""
        if request.messages:
            last_message = request.messages[-1]
            if isinstance(last_message.content, str):
                user_message_content = last_message.content
            elif isinstance(last_message.content, list):
                for part in reversed(last_message.content):
                    if hasattr(part, 'type') and part.type == 'text' and hasattr(part, 'text'):
                        user_message_content = part.text
                        break
        if not user_message_content:
            self.logger.warning("⚠️ 无法从请求中找到用户消息内容用于签名，将使用空字符串")

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
        url = f"{self.base_url}/api/chat/completions"

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

        # 6. 处理消息格式
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                # 处理多模态内容
                content_parts = []
                for part in msg.content:
                    if hasattr(part, 'type') and hasattr(part, 'text'):
                        content_parts.append({"type": part.type, "text": part.text})
                messages.append({"role": msg.role, "content": content_parts})

        # 7. 确定模型特性和上游模型ID
        requested_model = request.model
        is_search = "-search" in requested_model.casefold()
        # 判断思考模式，如果body中的messages中的content是list，就开启思考。
        is_anthropic_messages = any(isinstance(msg.content, list) for msg in request.messages)
        if is_anthropic_messages:
            self.logger.info(f"content为List，开启思考： {is_anthropic_messages}")
        # 正常判断，body中带thinking或者模型名带-thinking
        requested_thinking_enable = isinstance(request.thinking, dict) and request.thinking.get("type") == "enabled"
        is_thinking = is_anthropic_messages or requested_thinking_enable or ("-thinking" in requested_model.casefold())

        # 获取上游模型ID
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")

        # 8. 构建MCP服务器列表
        mcp_servers = []
        if is_search and "-4.5" in requested_model:
            mcp_servers.append("deep-web-search")
            self.logger.info("🔍 检测到搜索模型，添加 deep-web-search MCP 服务器")

        # 9. 构建上游请求体
        chat_id = self._generate_uuid()

        body = {
            "stream": True,  # 总是使用流式
            "model": upstream_model_id,
            "messages": messages,
            "signature_prompt": user_message_content,
            "params": {},
            "features": {
                "image_generation": False,
                "web_search": is_search,
                "auto_web_search": is_search,
                "preview_mode": False,
                "flags": [],
                "features": [
                    {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
                    {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
                    {"type": "mcp", "server": "image-search", "status": "hidden"},
                    {"type": "mcp", "server": "deep-research", "status": "hidden"},
                    {"type": "tool_selector", "server": "tool_selector", "status": "hidden"},
                    {"type": "mcp", "server": "advanced-search", "status": "hidden"},
                ],
                "enable_thinking": is_thinking,
            },
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
            },
            "mcp_servers": mcp_servers,
            "variables": {
                "{{USER_NAME}}": "Guest",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}": datetime.now().strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}": datetime.now().strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}": datetime.now().strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
                "{{USER_LANGUAGE}}": "zh-CN",
            },
            "model_item": {"id": upstream_model_id, "name": requested_model, "owned_by": "z.ai"},
            "chat_id": chat_id,
            "id": self._generate_uuid(),
        }

        # 处理工具支持
        if settings.TOOL_SUPPORT and not is_thinking and request.tools:
            body["tools"] = request.tools
            self.logger.info(f"启用工具支持: {len(request.tools)} 个工具")
        else:
            body["tools"] = None

        # 处理其他参数
        if request.temperature is not None:
            body["params"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["params"]["max_tokens"] = request.max_tokens

        # 9. 返回转换后的请求对象
        # 存储当前token用于错误处理
        self._current_token = token
        # 日志输出：格式化的json
        self.logger.debug(f"转换后的请求头:\n {json.dumps(headers, ensure_ascii=False, indent=2)}")
        self.logger.debug(f"转换后的请求参数:\n {json.dumps(params, ensure_ascii=False, indent=2)}")
        self.logger.debug(f"转换后的请求体:\n {json.dumps(body, ensure_ascii=False, indent=2)}")
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
                # transform_response 知道如何处理上游的流并聚合成单个响应
                return await self.transform_response(response, request, transformed)

        except Exception as e:
            self.log_response(False, str(e))
            if request.stream:
                # 对于流式请求，在主函数中捕获异常时返回一个错误生成器
                return self._streaming_error_generator(f"请求处理时发生未知错误: {e}", "internal_error")
            else:
                return self.handle_error(e, "请求处理")

    async def _create_stream_response(
        self, request: OpenAIRequest, transformed: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式响应生成器"""
        current_token = transformed.get("token", "")

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
                        error_msg = error_text.decode('utf-8', errors='ignore')
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
                        return

                    # 标记token使用成功（如果不是匿名模式）
                    if current_token and not settings.ANONYMOUS_MODE:
                        token_pool = get_token_pool()
                        if token_pool:
                            token_pool.mark_token_success(current_token)

                    # 处理流式响应
                    chat_id = transformed["chat_id"]
                    model = transformed["model"]
                    async for chunk in self._handle_stream_response(response, chat_id, model, transformed):
                        yield chunk
                    return

        except Exception as e:
            self.logger.error(f"❌ 流处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            # 标记token失败（如果不是匿名模式）
            if current_token and not settings.ANONYMOUS_MODE:
                self.mark_token_failure(current_token, e)

            # 返回错误
            error_response = {"error": {"message": str(e), "type": "stream_error"}}
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
            return

    async def transform_response(
        self, response: httpx.Response, request: OpenAIRequest, transformed: Dict[str, Any]
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
                return self._streaming_error_generator(error_msg, "upstream_error", response.status_code)

        if request.stream:
            return self._handle_stream_response(response, chat_id, model, transformed)
        else:
            return await self._handle_non_stream_response(response, chat_id, model)

    async def _handle_stream_response(
        self, response: httpx.Response, chat_id: str, model: str, transformed: Dict[str, Any]
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
            self.logger.info(f"🔧 初始化工具处理器: {len(transformed['body'].get('tools', []))} 个工具")

        has_thinking = False
        thinking_signature = None
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

                                if phase and phase != getattr(self, '_last_phase', None):
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
                                    for output in tool_handler.process_sse_chunk(sse_chunk):
                                        yield output
                                elif phase == "thinking":
                                    if not has_thinking:
                                        has_thinking = True
                                        role_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant"})
                                        yield await self.format_sse_chunk(role_chunk)

                                    delta_content = data.get("delta_content", "")
                                    if delta_content:
                                        # 处理思考内容格式
                                        content = (
                                            delta_content.split("</summary>\n>")[-1].strip()
                                            if delta_content.startswith("<details")
                                            and "</summary>\n>" in delta_content
                                            else delta_content
                                        )

                                        thinking_chunk = self.create_openai_chunk(
                                            chat_id,
                                            model,
                                            {"role": "assistant", "reasoning_content": content.replace("\n>", "\n")},
                                        )
                                        yield await self.format_sse_chunk(thinking_chunk)
                                elif phase == "answer":
                                    edit_content = data.get("edit_content", "")
                                    delta_content = data.get("delta_content", "")

                                    if edit_content and "</details>\n" in edit_content:
                                        if has_thinking:
                                            thinking_signature = str(int(time.time() * 1000))
                                            sig_chunk = self.create_openai_chunk(
                                                chat_id,
                                                model,
                                                {
                                                    "role": "assistant",
                                                    "thinking": {"content": "", "signature": thinking_signature},
                                                },
                                            )
                                            yield await self.format_sse_chunk(sig_chunk)

                                        content_after = edit_content.split("</details>\n")[-1]
                                        if content_after:
                                            content_chunk = self.create_openai_chunk(
                                                chat_id, model, {"role": "assistant", "content": content_after}
                                            )
                                            yield await self.format_sse_chunk(content_chunk)
                                    elif delta_content:
                                        if not has_thinking:
                                            has_thinking = True  # Mark as true to prevent sending role chunk again
                                            role_chunk = self.create_openai_chunk(
                                                chat_id, model, {"role": "assistant"}
                                            )
                                            yield await self.format_sse_chunk(role_chunk)

                                        content_chunk = self.create_openai_chunk(
                                            chat_id, model, {"role": "assistant", "content": delta_content}
                                        )
                                        yield await self.format_sse_chunk(content_chunk)

                                    if data.get("usage"):
                                        self.logger.info(f"📦 完成响应 - 使用统计: {json.dumps(data['usage'])}")
                                        if not tool_handler:
                                            finish_chunk = self.create_openai_chunk(
                                                chat_id, model, {"role": "assistant", "content": ""}, "stop"
                                            )
                                            finish_chunk["usage"] = data["usage"]
                                            yield await self.format_sse_chunk(finish_chunk)
                                            yield "data: [DONE]\n\n"
                        except json.JSONDecodeError as e:
                            self.logger.debug(f"❌ JSON解析错误: {e}, 内容: {chunk_str[:1000]}")
                        except Exception as e:
                            self.logger.error(f"❌ 处理chunk错误: {e}, chunk: {chunk_str[:1000]}")

            if not tool_handler:
                self.logger.debug("📤 发送最终 [DONE] 信号 (如果尚未发送)")
                # The final DONE is now sent within the usage block to ensure it's the very last thing for non-tool responses.

            self.logger.info(f"✅ SSE 流处理完成，共处理 {line_count} 行数据")

        except Exception as e:
            self.logger.error(f"❌ 流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            async for chunk in self._streaming_error_generator("流处理失败", "stream_error"):
                yield chunk

    async def _handle_non_stream_response(self, response: httpx.Response, chat_id: str, model: str) -> Dict[str, Any]:
        """处理非流式响应"""
        final_content = ""
        reasoning_content = ""
        usage_info: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            async for line in response.aiter_lines():
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
                        cleaned = (
                            delta_content.split("</summary>\n>")[-1].strip()
                            if delta_content.startswith("<details") and "</summary>\n>" in delta_content
                            else delta_content
                        )
                        reasoning_content += cleaned
                elif phase == "answer":
                    if edit_content and "</details>\n" in edit_content:
                        final_content += edit_content.split("</details>\n")[-1]
                    elif delta_content:
                        final_content += delta_content

        except Exception as e:
            self.logger.error(f"❌ 非流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return self.handle_error(e, "非流式聚合")

        final_content = (final_content or "").strip()
        reasoning_content = (reasoning_content or "").strip()
        if not final_content and reasoning_content:
            final_content = reasoning_content

        return self.create_openai_response_with_reasoning(chat_id, model, final_content, reasoning_content, usage_info)
