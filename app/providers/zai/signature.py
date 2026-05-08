#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI JWT 签名生成
"""

import json
import time
import hmac
import hashlib
import base64
import uuid as _uuid
from typing import Dict, Any, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger()


def generate_uuid() -> str:
    return str(_uuid.uuid4())


def generate_signature_params(
    token: str, user_message: Optional[str]
) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.error("无效的JWT令牌格式，无法生成签名")
            return None

        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        decoded_payload = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(decoded_payload)
        user_id = payload.get("id")
        if not user_id:
            logger.error("从JWT令牌中无法获取 'id'，无法生成签名")
            return None

        timestamp = int(time.time() * 1000)
        request_id = generate_uuid()
        chat_id = generate_uuid()

        time_5min_split = timestamp // (5 * 60 * 1000)
        key = settings.ZAI_SIGNATURE_KEY.encode("utf-8")
        signature_pre = hmac.new(
            key, str(time_5min_split).encode("utf-8"), hashlib.sha256
        ).hexdigest()

        safe_user_message = user_message or ""
        request_info = (
            f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
        )
        user_message_encode = base64.b64encode(
            safe_user_message.encode("utf-8")
        ).decode("utf-8")
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
        logger.error(f"生成签名参数时出错: {e}")
        return None
