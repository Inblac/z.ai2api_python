#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import uuid
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator
import httpx
import asyncio

from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.token_pool import get_token_pool, initialize_token_pool
from app.utils.user_agent import get_random_user_agent

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
        "X-FE-Version": "prod-fe-1.0.98",
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


def generate_uuid() -> str:
    """生成UUID v4"""
    return str(uuid.uuid4())


def get_auth_token_sync() -> str:
    """同步获取认证令牌（用于非异步场景）"""
    # 如果启用匿名模式，只尝试获取访客令牌
    if settings.ANONYMOUS_MODE:
        try:
            headers = get_zai_dynamic_headers()
            with httpx.Client() as client:
                response = client.get("https://chat.z.ai/api/v1/auths/", headers=headers, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    token = data.get("token", "")
                    if token:
                        logger.debug(f"获取访客令牌成功: {token[:20]}...")
                        return token
        except Exception as e:
            logger.warning(f"获取访客令牌失败: {e}")

        # 匿名模式下，如果获取访客令牌失败，直接返回空
        logger.error("❌ 匿名模式下获取访客令牌失败")
        return ""

    # 非匿名模式：首先使用token池获取备份令牌
    token_pool = get_token_pool()
    if token_pool:
        token = token_pool.get_next_token()
        if token:
            logger.debug(f"从token池获取令牌: {token[:20]}...")
            return token

    # 如果没有备份token，尝试降级到匿名模式
    logger.warning("⚠️ 没有可用的备份token，尝试降级到匿名模式...")
    try:
        headers = get_zai_dynamic_headers()
        with httpx.Client() as client:
            response = client.get("https://chat.z.ai/api/v1/auths/", headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                token = data.get("token", "")
                if token:
                    logger.info(f"✅ 降级到匿名模式成功: {token[:20]}...")
                    return token
    except Exception as e:
        logger.warning(f"降级到匿名模式失败: {e}")

    # 没有可用的token
    logger.error("❌ 所有认证方式都失败了")
    return ""
