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
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from app.providers.base import BaseProvider, ProviderConfig
from app.models.schemas import OpenAIRequest
from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.token_pool import get_token_pool
from app.core.zai_transformer import generate_uuid, get_zai_dynamic_headers
from app.utils.sse_tool_handler import SSEToolHandler

logger = get_logger()


class ZAIProvider(BaseProvider):
    """Z.AI 提供商"""
    
    def __init__(self):
        config = ProviderConfig(
            name="zai",
            api_endpoint=settings.API_ENDPOINT,
            timeout=30,
            headers=get_zai_dynamic_headers()
        )
        super().__init__(config)
        
        # Z.AI 特定配置
        self.base_url = "https://chat.z.ai"
        self.auth_url = f"{self.base_url}/api/v1/auths/"
        
        # 存储客户端传递的 token
        self._client_token = None
        
        # 模型映射
        self.model_mapping = {
            settings.PRIMARY_MODEL: "0727-360B-API",  # GLM-4.5
            settings.THINKING_MODEL: "0727-360B-API",  # GLM-4.5-Thinking
            settings.SEARCH_MODEL: "0727-360B-API",  # GLM-4.5-Search
            settings.AIR_MODEL: "0727-106B-API",  # GLM-4.5-Air
            settings.GLM46_MODEL: "GLM-4-6-API-V1",  # GLM-4.6
            settings.GLM46_THINKING_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Thinking
            settings.GLM46_SEARCH_MODEL: "GLM-4-6-API-V1",  # GLM-4.6-Search
        }
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            settings.PRIMARY_MODEL,
            settings.THINKING_MODEL,
            settings.SEARCH_MODEL,
            settings.AIR_MODEL,
            settings.GLM46_MODEL,
            settings.GLM46_THINKING_MODEL,
            settings.GLM46_SEARCH_MODEL,
        ]
    
    async def get_token(self) -> str:
        """获取认证令牌"""
        # 如果启用客户端 token 模式且有客户端 token，优先使用
        if settings.USE_CLIENT_TOKEN and self._client_token and not settings.ANONYMOUS_MODE:
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

        # 如果token池为空或没有可用token，使用配置的AUTH_TOKEN
        if settings.AUTH_TOKEN and settings.AUTH_TOKEN != "sk-your-api-key":
            self.logger.debug("使用配置的AUTH_TOKEN")
            return settings.AUTH_TOKEN

        self.logger.error("❌ 无法获取有效的认证令牌")
        return ""
    
    def mark_token_failure(self, token: str, error: Optional[Exception] = None):
        """标记token使用失败"""
        token_pool = get_token_pool()
        if token_pool:
            error_to_report = error if error is not None else Exception("Token failure reported without specific error")
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
            request_id = str(uuid.uuid4())
            chat_id = str(uuid.uuid4())

            # 3. 构造签名
            safe_user_message = user_message or ""
            e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
            i = f"{e}|{safe_user_message}|{str(timestamp)}"
            n = timestamp // (5 * 60 * 1000)
            key = "junjie".encode('utf-8')
            o = hmac.new(key, str(n).encode('utf-8'), hashlib.sha256).hexdigest()
            signature = hmac.new(o.encode('utf-8'), i.encode('utf-8'), hashlib.sha256).hexdigest()

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
    
    async def transform_request(self, request: OpenAIRequest) -> Dict[str, Any]:
        """转换OpenAI请求为Z.AI格式"""
        self.logger.info(f"🔄 转换 OpenAI 请求到 Z.AI 格式: {request.model}")

        # 1. 获取认证令牌
        token = await self.get_token()
        if not token:
            self.logger.error("❌ 无法获取认证令牌，请求将失败")
            return {
                "url": self.config.api_endpoint, "params": {}, "headers": {}, "body": {}, "token": None,
                "chat_id": "no-chat-id", "model": request.model
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
                "url": self.config.api_endpoint, "params": {}, "headers": {}, "body": {}, "token": token,
                "chat_id": "no-chat-id", "model": request.model
            }

        user_id = signature_params["user_id"]
        chat_id = signature_params["chat_id"]
        request_id = signature_params["request_id"]
        timestamp = signature_params["timestamp"]
        signature = signature_params["signature"]

        # 4. 构建请求 URL 和 Params
        url = f"{self.base_url}/api/chat/completions"
        params = {
            "timestamp": timestamp,
            "requestId": request_id,
            "user_id": user_id,
            "token": token,
            "current_url": f"{self.base_url}/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "signature_timestamp": timestamp,
        }

        # 5. 构建请求头
        headers = get_zai_dynamic_headers(chat_id)
        headers["Authorization"]= f"Bearer {token}"
        headers["X-Signature"]= signature
        

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
        is_thinking = "-thinking" in requested_model.casefold()
        is_search = "-search" in requested_model.casefold()
        
        # 获取上游模型ID
        upstream_model_id = self.model_mapping.get(requested_model, "0727-360B-API")

        # 8. 构建MCP服务器列表
        mcp_servers = []
        if is_search and "-4.5" in requested_model:
            mcp_servers.append("deep-web-search")
            self.logger.info("🔍 检测到搜索模型，添加 deep-web-search MCP 服务器")
        
        # 9. 构建上游请求体
        chat_id = generate_uuid()
        
        body = {
            "stream": True,  # 总是使用流式
            "model": upstream_model_id,
            "messages": messages,
            "params": {},
            "features": {
                "image_generation": False,
                "web_search": is_search,
                "auto_web_search": is_search,
                "preview_mode": False,
                "flags": [],
                "features": [
                    {
                        "type": "mcp",
                        "server": "vibe-coding",
                        "status": "hidden"
                    },
                    {
                        "type": "mcp",
                        "server": "ppt-maker",
                        "status": "hidden"
                    },
                    {
                        "type": "mcp",
                        "server": "image-search",
                        "status": "hidden"
                    },
                    {
                        "type": "mcp",
                        "server": "deep-research",
                        "status": "hidden"
                    },
                    {
                        "type": "tool_selector",
                        "server": "tool_selector",
                        "status": "hidden"
                    },
                    {
                        "type": "mcp",
                        "server": "advanced-search",
                        "status": "hidden"
                    }
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
            "model_item": {
                "id": upstream_model_id,
                "name": requested_model,
                "owned_by": "z.ai"
            },
            "chat_id": chat_id,
            "id": generate_uuid(),
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
        return {
            "url": url,
            "params": params,
            "headers": headers,
            "body": body,
            "token": token,
            "chat_id": chat_id,
            "model": requested_model
        }
    
    @staticmethod
    async def _streaming_error_generator(error_msg: str, error_type: str, code: Optional[int] = None) -> AsyncGenerator[str, None]:
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
                return self._create_stream_response_with_retry(request, transformed)
            else:
                # 非流式响应
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        transformed["url"],
                        params=transformed.get("params", {}),
                        headers=transformed["headers"],
                        json=transformed["body"]
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

    async def _create_stream_response_with_retry(
        self,
        request: OpenAIRequest,
        transformed: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """创建带重试机制的流式响应生成器"""
        retry_count = 0
        last_error = None

        current_transformed = transformed

        while retry_count <= settings.MAX_RETRIES:
            current_token = current_transformed.get("token", "")
            try:
                if retry_count > 0:
                    delay = settings.RETRY_DELAY
                    self.logger.warning(f"重试请求 ({retry_count}/{settings.MAX_RETRIES}) - 等待 {delay:.1f}s")
                    await asyncio.sleep(delay)

                    if current_token and not settings.ANONYMOUS_MODE:
                        self.mark_token_failure(current_token, Exception(f"Retry {retry_count}: {last_error}"))

                    self.logger.info("🔑 重新转换请求用于重试...")
                    current_transformed = await self.transform_request(request)
                    new_token = current_transformed.get("token", "")
                    if not new_token:
                        raise Exception("重试时无法获取有效的认证令牌")
                    current_token = new_token

                async with httpx.AsyncClient(timeout=60.0) as client:
                    self.logger.info(f"🎯 发送请求到 Z.AI: {current_transformed['url']} with params {current_transformed.get('params', {})}")
                    async with client.stream(
                        "POST",
                        current_transformed["url"],
                        params=current_transformed.get("params", {}),
                        json=current_transformed["body"],
                        headers=current_transformed["headers"],
                    ) as response:
                        if response.status_code == 400:
                            error_text = await response.aread()
                            error_msg = error_text.decode('utf-8', errors='ignore')
                            self.logger.warning(f"❌ 上游返回 400 错误 (尝试 {retry_count + 1}/{settings.MAX_RETRIES + 1})")
                            retry_count += 1
                            last_error = f"400 Bad Request: {error_msg}"
                            if retry_count > settings.MAX_RETRIES:
                                self.logger.error(f"❌ 达到最大重试次数 ({settings.MAX_RETRIES})，请求失败")
                                async for chunk in self._streaming_error_generator(
                                    f"Request failed after {settings.MAX_RETRIES} retries: {last_error}", "upstream_error", 400
                                ):
                                    yield chunk
                                return
                            continue

                        elif response.status_code != 200:
                            error_text = await response.aread()
                            error_msg = error_text.decode('utf-8', errors='ignore')
                            self.logger.error(f"❌ 上游返回错误: {response.status_code}, 详情: {error_msg}")
                            async for chunk in self._streaming_error_generator(
                                f"Upstream error: {response.status_code}", "upstream_error", response.status_code
                            ):
                                yield chunk
                            return

                        if retry_count > 0:
                            self.logger.info(f"✨ 第 {retry_count} 次重试成功")

                        if current_token and not settings.ANONYMOUS_MODE:
                            token_pool = get_token_pool()
                            if token_pool:
                                token_pool.mark_token_success(current_token)

                        chat_id = current_transformed["chat_id"]
                        model = current_transformed["model"]
                        async for chunk in self._handle_stream_response(response, chat_id, model, current_transformed):
                            yield chunk
                        return

            except Exception as e:
                self.logger.error(f"❌ 流处理错误: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

                if current_token and not settings.ANONYMOUS_MODE:
                    self.mark_token_failure(current_token, e)

                retry_count += 1
                last_error = str(e)

                if retry_count > settings.MAX_RETRIES:
                    self.logger.error(f"❌ 达到最大重试次数 ({settings.MAX_RETRIES})，流处理失败")
                    async for chunk in self._streaming_error_generator(
                        f"Stream processing failed after {settings.MAX_RETRIES} retries: {last_error}", "stream_error"
                    ):
                        yield chunk
                    return

    async def transform_response(
        self,
        response: httpx.Response,
        request: OpenAIRequest,
        transformed: Dict[str, Any]
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
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        transformed: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """处理Z.AI流式响应"""
        self.logger.info(f"✅ Z.AI 响应成功，开始处理 SSE 流")

        has_tools = transformed["body"].get("tools") is not None
        tool_handler = None
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

                        self.logger.debug(f"📦 解析数据块: {chunk_str[:1000]}..." if len(chunk_str) > 1000 else f"📦 解析数据块: {chunk_str}")

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
                                        "phase": phase, "edit_content": data.get("edit_content", ""),
                                        "delta_content": data.get("delta_content", ""), "edit_index": data.get("edit_index"),
                                        "usage": data.get("usage", {})
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
                                        content = delta_content.split("</summary>\n>")[-1].strip() if delta_content.startswith("<details") and "</summary>\n>" in delta_content else delta_content
                                        thinking_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant", "reasoning_content":  content.replace("\n>","\n")})
                                        yield await self.format_sse_chunk(thinking_chunk)
                                elif phase == "answer":
                                    edit_content = data.get("edit_content", "")
                                    delta_content = data.get("delta_content", "")

                                    if edit_content and "</details>\n" in edit_content:
                                        if has_thinking:
                                            thinking_signature = str(int(time.time() * 1000))
                                            sig_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant", "thinking": {"content": "", "signature": thinking_signature}})
                                            yield await self.format_sse_chunk(sig_chunk)

                                        content_after = edit_content.split("</details>\n")[-1]
                                        if content_after:
                                            content_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant", "content": content_after})
                                            yield await self.format_sse_chunk(content_chunk)
                                    elif delta_content:
                                        if not has_thinking:
                                            has_thinking = True # Mark as true to prevent sending role chunk again
                                            role_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant"})
                                            yield await self.format_sse_chunk(role_chunk)

                                        content_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant", "content": delta_content})
                                        yield await self.format_sse_chunk(content_chunk)

                                    if data.get("usage"):
                                        self.logger.info(f"📦 完成响应 - 使用统计: {json.dumps(data['usage'])}")
                                        if not tool_handler:
                                            finish_chunk = self.create_openai_chunk(chat_id, model, {"role": "assistant", "content": ""}, "stop")
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

    async def _handle_non_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str
    ) -> Dict[str, Any]:
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
                        cleaned = delta_content.split("</summary>\n>")[-1].strip() if delta_content.startswith("<details") and "</summary>\n>" in delta_content else delta_content
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

        return self.create_openai_response_with_reasoning(
            chat_id, model, final_content, reasoning_content, usage_info
        )
