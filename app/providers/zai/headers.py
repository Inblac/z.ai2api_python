#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 动态 HTTP headers 和浏览器指纹
"""

import random
from datetime import datetime
from typing import Dict, Any

from app.core.config import settings
from app.utils.user_agent import get_random_user_agent


def get_zai_dynamic_headers(chat_id: str = "") -> Dict[str, str]:
    browser_choices = [
        "chrome", "chrome", "chrome",
        "edge", "edge",
        "firefox",
        "safari",
    ]
    browser_type = random.choice(browser_choices)
    user_agent = get_random_user_agent(browser_type)

    chrome_version = "139"
    edge_version = "139"
    sec_ch_ua = None

    if "Chrome/" in user_agent:
        try:
            chrome_version = user_agent.split("Chrome/")[1].split(".")[0]
            sec_ch_ua = f'"Google Chrome";v="{chrome_version}", "Not?A_Brand";v="8", "Chromium";v="{chrome_version}"'
        except Exception:
            pass

    if "Edg/" in user_agent:
        try:
            edge_version = user_agent.split("Edg/")[1].split(".")[0]
            sec_ch_ua = f'"Microsoft Edge";v="{edge_version}", "Chromium";v="{chrome_version}", "Not_A Brand";v="24"'
        except Exception:
            sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}"'
    elif "Firefox/" in user_agent:
        sec_ch_ua = None
    else:
        sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}"'

    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": user_agent,
        "Accept-Language": "en-US",
        "DNT": "1",
        "Priority": "u=1, i",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-FE-Version": settings.X_FE_VERSION,
        "X-Region": "domestic",
        "Origin": "https://chat.z.ai",
    }

    if sec_ch_ua:
        headers["sec-ch-ua"] = sec_ch_ua
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'

    if chat_id:
        headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        headers["Referer"] = "https://chat.z.ai/"

    return headers


def generate_browser_params(user_agent: str) -> Dict[str, Any]:
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
        "language": "en-US",
        "languages": "en-US",
        "timezone": "Asia/Shanghai",
        "timezone_offset": -480,
        "local_time": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "utc_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "cookie_enabled": "true",
        "screen_width": "1920",
        "screen_height": "1080",
        "screen_resolution": "1920x1080",
        "viewport_width": "1920",
        "viewport_height": "869",
        "viewport_size": "1920x869",
        "color_depth": "32",
        "pixel_ratio": "1",
        "is_mobile": "false",
        "is_touch": "false",
        "max_touch_points": "40",
        "search": "",
        "hash": "",
        "host": "chat.z.ai",
        "hostname": "chat.z.ai",
        "protocol": "https:",
        "referrer": "",
        "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
        "user_agent": user_agent,
        "browser_name": browser_name,
        "os_name": os_name,
    }
