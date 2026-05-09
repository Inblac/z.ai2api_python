#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 上游聊天会话 CRUD
"""

import time
import httpx
from typing import Dict, List, Optional

from app.providers.zai.signature import generate_uuid
from app.utils.logger import get_logger

logger = get_logger()

BASE_URL = "https://chat.z.ai"


async def create_upstream_chat(
    prompt: str,
    upstream_model_id: str,
    token: str,
    user_agent: str,
    enable_thinking: bool = False,
    web_search: bool = False,
    user_message_id: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
) -> str:
    message_id = user_message_id or generate_uuid()
    timestamp_seconds = int(time.time())
    body = {
        "chat": {
            "id": "",
            "title": "新聊天",
            "models": [upstream_model_id],
            "params": {},
            "history": {
                "messages": {
                    message_id: {
                        "id": message_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": prompt or "",
                        "timestamp": timestamp_seconds,
                        "models": [upstream_model_id],
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
        "User-Agent": user_agent,
        "Accept-Language": "en-US",
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/chats/new",
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


async def delete_upstream_chat(
    chat_id: str,
    token: str,
    headers: Dict[str, str],
) -> None:
    if not chat_id or not token:
        logger.warning(
            "跳过删除上游 chat，参数缺失: chat_id={}, token={}",
            chat_id or "空", "有" if token else "空",
        )
        return

    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": headers.get("User-Agent", ""),
        "Accept-Language": headers.get("Accept-Language", "en-US,zh;q=0.9,en-US;q=0.8,en;q=0.7"),
        "Cache-Control": "no-cache",
        "DNT": headers.get("DNT", "1"),
        "Priority": headers.get("Priority", "u=1, i"),
        "sec-ch-ua": headers.get("sec-ch-ua", ""),
        "sec-ch-ua-mobile": headers.get("sec-ch-ua-mobile", "?0"),
        "sec-ch-ua-platform": headers.get("sec-ch-ua-platform", '"Windows"'),
        "Sec-Fetch-Dest": headers.get("Sec-Fetch-Dest", "empty"),
        "Sec-Fetch-Mode": headers.get("Sec-Fetch-Mode", "cors"),
        "Sec-Fetch-Site": headers.get("Sec-Fetch-Site", "same-origin"),
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/c/{chat_id}",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{BASE_URL}/api/v1/chats/{chat_id}",
                headers=request_headers,
            )

        if response.is_success:
            logger.info(f"已删除上游 chat: {chat_id}")
        else:
            logger.warning(
                "删除上游 chat 失败: {} {}", response.status_code, response.text
            )
    except Exception as e:
        logger.warning(f"删除上游 chat 异常: {e}")


async def list_upstream_chats(token: str, headers: Dict[str, str]) -> list[dict]:
    """获取上游所有 default 类型的对话列表。"""
    if not token:
        logger.warning("跳过获取对话列表，token 为空")
        return []

    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": headers.get("User-Agent", ""),
        "Accept-Language": headers.get("Accept-Language", "en-US,zh;q=0.9,en-US;q=0.8,en;q=0.7"),
        "Priority": headers.get("Priority", "u=1, i"),
        "sec-ch-ua": headers.get("sec-ch-ua", ""),
        "sec-ch-ua-mobile": headers.get("sec-ch-ua-mobile", "?0"),
        "sec-ch-ua-platform": headers.get("sec-ch-ua-platform", '"Windows"'),
        "Sec-Fetch-Dest": headers.get("Sec-Fetch-Dest", "empty"),
        "Sec-Fetch-Mode": headers.get("Sec-Fetch-Mode", "cors"),
        "Sec-Fetch-Site": headers.get("Sec-Fetch-Site", "same-origin"),
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{BASE_URL}/api/v1/chats/",
                params={"page": 1, "type": "default"},
                headers=request_headers,
            )

        if response.is_success:
            chats: list[dict] = response.json()
            logger.info("获取到 {} 个上游对话", len(chats))
            return chats
        else:
            logger.warning("获取对话列表失败: {} {}", response.status_code, response.text)
            return []
    except Exception as e:
        logger.warning(f"获取对话列表异常: {e}")
        return []


async def delete_all_upstream_chats(
    token: str,
    headers: Dict[str, str],
) -> None:
    """获取所有对话列表，逐个全部删除。"""
    if not token:
        logger.warning("跳过批量删除，token 为空")
        return

    chats = await list_upstream_chats(token, headers)
    for chat in chats:
        chat_id = chat.get("id", "")
        await delete_upstream_chat(chat_id, token, headers)

    logger.info("批量删除完成: 共删除 {} 个对话", len(chats))
