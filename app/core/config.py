#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_ENDPOINT: str = "https://chat.z.ai/api/chat/completions"
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "sk-your-api-key")

    # Model Configuration
    GLM45_MODEL: str = os.getenv("GLM45_MODEL", "GLM-4.5")
    GLM45_THINKING_MODEL: str = os.getenv("GLM45_THINKING_MODEL", "GLM-4.5-Thinking")
    GLM45_SEARCH_MODEL: str = os.getenv("GLM45_SEARCH_MODEL", "GLM-4.5-Search")
    AIR_MODEL: str = os.getenv("AIR_MODEL", "GLM-4.5-Air")
    GLM46_MODEL: str = os.getenv("GLM46_MODEL", "GLM-4.6")
    GLM46_THINKING_MODEL: str = os.getenv("GLM46_THINKING_MODEL", "GLM-4.6-Thinking")
    GLM46_SEARCH_MODEL: str = os.getenv("GLM46_SEARCH_MODEL", "GLM-4.6-Search")
    GLM47_MODEL: str = os.getenv("GLM47_MODEL", "GLM-4.7")
    GLM47_THINKING_MODEL: str = os.getenv("GLM47_THINKING_MODEL", "GLM-4.7-Thinking")
    GLM47_SEARCH_MODEL: str = os.getenv("GLM47_SEARCH_MODEL", "GLM-4.7-Search")
    GLM5_MODEL: str = os.getenv("GLM5_MODEL", "GLM-5")
    GLM5_THINKING_MODEL: str = os.getenv("GLM5_THINKING_MODEL", "GLM-5-Thinking")
    GLM5_SEARCH_MODEL: str = os.getenv("GLM5_SEARCH_MODEL", "GLM-5-Search")
    GLM5T_MODEL: str = os.getenv("GLM5T_MODEL", "GLM-5-Turbo")
    GLM5T_THINKING_MODEL: str = os.getenv(
        "GLM5T_THINKING_MODEL", "GLM-5-Turbo-Thinking"
    )
    GLM5T_SEARCH_MODEL: str = os.getenv("GLM5T_SEARCH_MODEL", "GLM-5-Turbo-Search")
    GLM51_MODEL: str = os.getenv("GLM51_MODEL", "GLM-5.1")
    GLM51_THINKING_MODEL: str = os.getenv("GLM51_THINKING_MODEL", "GLM-5.1-Thinking")
    GLM51_SEARCH_MODEL: str = os.getenv("GLM51_SEARCH_MODEL", "GLM-5.1-Search")

    # Provider Model Mapping
    @property
    def provider_model_mapping(self) -> Dict[str, str]:
        """模型到提供商的映射"""
        return {
            # Z.AI models
            "GLM-4.5": "zai",
            "GLM-4.5-Thinking": "zai",
            "GLM-4.5-Search": "zai",
            "GLM-4.5-Air": "zai",
            "GLM-4.6": "zai",
            "GLM-4.6-Thinking": "zai",
            "GLM-4.6-Search": "zai",
            "GLM-4.7": "zai",
            "GLM-4.7-Thinking": "zai",
            "GLM-4.7-Search": "zai",
            "GLM-5": "zai",
            "GLM-5-Thinking": "zai",
            "GLM-5-Search": "zai",
            "GLM-5-Turbo": "zai",
            "GLM-5-Turbo-Thinking": "zai",
            "GLM-5-Turbo-Search": "zai",
            "GLM-5.1": "zai",
            "GLM-5.1-Thinking": "zai",
            "GLM-5.1-Search": "zai",
        }

    # Server Configuration
    LISTEN_PORT: int = int(os.getenv("LISTEN_PORT", "8080"))
    DEBUG_LOGGING: bool = os.getenv("DEBUG_LOGGING", "true").lower() == "true"
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "z-ai2api-server")

    ANONYMOUS_MODE: bool = os.getenv("ANONYMOUS_MODE", "true").lower() == "true"
    TOOL_SUPPORT: bool = os.getenv("TOOL_SUPPORT", "true").lower() == "true"
    SKIP_AUTH_TOKEN: bool = os.getenv("SKIP_AUTH_TOKEN", "false").lower() == "true"

    # 是否使用客户端传递的 api_key 作为 Z.AI 认证 token
    USE_CLIENT_TOKEN: bool = os.getenv("USE_CLIENT_TOKEN", "false").lower() == "true"
    # 请求结束后是否自动删除 Z.AI 上游会话
    AUTO_DELETE_UPSTREAM_CHAT: bool = (
        os.getenv("AUTO_DELETE_UPSTREAM_CHAT", "true").lower() == "true"
    )

    # Z.AI Signature Configuration
    ZAI_SIGNATURE_KEY: str = os.getenv("ZAI_SIGNATURE_KEY", "junjie")

    # X-FE-Version Header
    X_FE_VERSION: str = os.getenv("X_FE_VERSION", "prod-fe-1.0.106")

    class Config:
        env_file = ".env"


settings = Settings()
