#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.config import settings
from app.models.schemas import OpenAIRequest, ModelsResponse, Model
from app.utils.logger import get_logger
from app.providers import get_provider_router

logger = get_logger()
router = APIRouter()

# 全局提供商路由器实例
provider_router = None


def get_provider_router_instance():
    """获取提供商路由器实例"""
    global provider_router
    if provider_router is None:
        provider_router = get_provider_router()
    return provider_router


@router.get("/v1/models")
async def list_models():
    """List available models from all providers"""
    try:
        router_instance = get_provider_router_instance()
        models_data = router_instance.get_models_list()
        return JSONResponse(content=models_data)
    except Exception as e:
        logger.error(f"❌ 获取模型列表失败: {e}")
        # 返回默认模型列表作为后备
        current_time = int(time.time())
        fallback_response = ModelsResponse(
            data=[
                Model(id=settings.GLM45_MODEL, created=current_time, owned_by="z.ai"),
                Model(id=settings.GLM45_THINKING_MODEL, created=current_time, owned_by="z.ai"),
                Model(id=settings.GLM45_SEARCH_MODEL, created=current_time, owned_by="z.ai"),
                Model(id=settings.AIR_MODEL, created=current_time, owned_by="z.ai"),
            ]
        )
        return fallback_response


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, authorization: str = Header(None)):
    """Handle chat completion requests with multi-provider architecture"""
    role = request.messages[0].role if request.messages else "unknown"
    logger.info(f"😶‍🌫️ 收到客户端请求 - 模型: {request.model}, 流式: {request.stream}, 消息数: {len(request.messages)}, 角色: {role}, 工具数: {len(request.tools) if request.tools else 0}")

    try:
        # 提取客户端 api_key（无论是否跳过认证）
        client_api_key = None
        if authorization and authorization.startswith("Bearer "):
            client_api_key = authorization[7:]

        # Validate API key (skip if SKIP_AUTH_TOKEN is enabled)
        if not settings.SKIP_AUTH_TOKEN:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

            if not client_api_key:
                raise HTTPException(status_code=401, detail="Missing API key")
            
            if client_api_key != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")

        # 使用多提供商路由器处理请求，传递客户端 api_key
        router_instance = get_provider_router_instance()
        result = await router_instance.route_request(request, client_api_key=client_api_key)

        # 检查是否有错误
        if isinstance(result, dict) and "error" in result:
            error_info = result["error"]
            if error_info.get("code") == "model_not_found":
                raise HTTPException(status_code=404, detail=error_info["message"])
            else:
                raise HTTPException(status_code=500, detail=error_info["message"])

        # 处理响应
        if request.stream:
            # 流式响应
            if hasattr(result, '__aiter__'):
                # 结果是异步生成器
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                    }
                )
            else:
                # 结果是字典，可能包含错误
                raise HTTPException(status_code=500, detail="Expected streaming response but got non-streaming result")
        else:
            # 非流式响应
            return JSONResponse(content=result)

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"❌ 请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
