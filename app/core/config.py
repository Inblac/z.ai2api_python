#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Configuration
    AUTH_TOKEN: str = "sk-your-api-key"

    # Server Configuration
    LISTEN_PORT: int = 8080
    DEBUG_LOGGING: bool = True
    SERVICE_NAME: str = "z-ai2api-server"

    ANONYMOUS_MODE: bool = True
    TOOL_SUPPORT: bool = True
    # 请求结束后是否批量删除账号下所有 Z.AI 上游 default 会话
    AUTO_DELETE_UPSTREAM_CHAT: bool = True

    # Z.AI Signature Configuration
    ZAI_SIGNATURE_KEY: str = "junjie"

    # X-FE-Version Header
    X_FE_VERSION: str = "prod-fe-1.0.106"

settings = Settings()
