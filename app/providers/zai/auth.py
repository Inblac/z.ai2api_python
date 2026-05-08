#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 认证令牌获取
"""

import httpx
from typing import Optional

from app.core.config import settings
from app.providers.zai.headers import get_zai_dynamic_headers
from app.utils.logger import get_logger

logger = get_logger()


async def get_zai_token(
    client_token: Optional[str] = None,
) -> str:
    if settings.USE_CLIENT_TOKEN and client_token:
        logger.debug(f"使用客户端传递的 token: {client_token[:20]}...")
        return client_token

    if settings.ANONYMOUS_MODE:
        try:
            headers = get_zai_dynamic_headers()
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://chat.z.ai/api/v1/auths/",
                    headers=headers,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    token = data.get("token", "")
                    if token:
                        logger.debug(f"获取访客令牌成功: {token[:20]}...")
                        return token
        except Exception as e:
            logger.warning(f"异步获取访客令牌失败: {e}")

        logger.error("匿名模式下获取访客令牌失败")
        return ""

    logger.error("无法获取有效的认证令牌")
    return ""
