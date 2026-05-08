#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户代理工具模块
提供动态随机用户代理生成功能
"""

import random
from typing import Optional
from fake_useragent import UserAgent

# 全局 UserAgent 实例（单例模式）
_user_agent_instance: Optional[UserAgent] = None


def get_user_agent_instance() -> UserAgent:
    """获取或创建 UserAgent 实例（单例模式）"""
    global _user_agent_instance
    if _user_agent_instance is None:
        _user_agent_instance = UserAgent(min_version=135)
    return _user_agent_instance


def get_random_user_agent(browser_type: Optional[str] = None) -> str:
    """
    获取随机用户代理字符串

    Args:
        browser_type: 指定浏览器类型 ('chrome', 'firefox', 'safari', 'edge')
                     如果为 None，则随机选择

    Returns:
        str: 用户代理字符串
    """
    ua = get_user_agent_instance()

    # 如果没有指定浏览器类型，随机选择一个（偏向 Chrome 和 Edge）
    if browser_type is None:
        browser_choices = ["chrome", "chrome", "chrome", "edge", "edge", "firefox", "safari"]
        browser_type = random.choice(browser_choices)

    # 根据浏览器类型获取用户代理
    if browser_type == "chrome":
        user_agent = ua.chrome
    elif browser_type == "edge":
        user_agent = ua.edge
    elif browser_type == "firefox":
        user_agent = ua.firefox
    elif browser_type == "safari":
        user_agent = ua.safari
    else:
        user_agent = ua.random

    return user_agent


