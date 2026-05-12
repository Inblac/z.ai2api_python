#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.config import settings
from app.models.schemas import OpenAIRequest, ModelsResponse, Model
from app.utils.logger import get_logger
from app.providers.zai.provider import ZAIProvider, SUPPORTED_MODELS

logger = get_logger()
router = APIRouter()

zai_provider = ZAIProvider()
supported_models_set = set(SUPPORTED_MODELS)


def resolve_client_zai_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    bearer_token = authorization[7:]
    if not bearer_token or bearer_token == settings.AUTH_TOKEN:
        return None
    return bearer_token


@router.get("/v1/models")
async def list_models():
    current_time = int(time.time())
    return ModelsResponse(
        data=[
            Model(id=m, created=current_time, owned_by="z.ai")
            for m in SUPPORTED_MODELS
        ]
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIRequest, authorization: str = Header(None)):
    role = request.messages[0].role if request.messages else "unknown"
    logger.info(
        "收到客户端请求 - 模型: {}, 流式: {}, 消息数: {}, 角色: {}, 工具数: {}",
        request.model, request.stream, len(request.messages), role,
        len(request.tools) if request.tools else 0,
    )

    if request.model not in supported_models_set:
        raise HTTPException(
            status_code=404,
            detail=f"不支持的模型: {request.model}",
        )

    try:
        client_zai_token = resolve_client_zai_token(authorization)

        result = await zai_provider.chat_completion(
            request, client_api_key=client_zai_token
        )

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"]["message"])

        if request.stream:
            if hasattr(result, "__aiter__"):
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                    },
                )
            raise HTTPException(
                status_code=500,
                detail="Expected streaming response but got non-streaming result",
            )
        else:
            return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("请求处理失败: {}", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )
