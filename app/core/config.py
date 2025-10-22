#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from app.utils.logger import logger


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_ENDPOINT: str = "https://chat.z.ai/api/chat/completions"
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "sk-your-api-key")

    # 认证token文件路径（可选）
    AUTH_TOKENS_FILE: Optional[str] = os.getenv("AUTH_TOKENS_FILE")

    # Token池配置
    TOKEN_HEALTH_CHECK_INTERVAL: int = int(os.getenv("TOKEN_HEALTH_CHECK_INTERVAL", "300"))  # 5分钟
    TOKEN_FAILURE_THRESHOLD: int = int(os.getenv("TOKEN_FAILURE_THRESHOLD", "3"))  # 失败3次后标记为不可用
    TOKEN_RECOVERY_TIMEOUT: int = int(os.getenv("TOKEN_RECOVERY_TIMEOUT", "1800"))  # 30分钟后重试失败的token

    def _load_tokens_from_file(self, file_path: str) -> List[str]:
        """
        从文件加载token列表

        支持多种格式的混合使用：
        1. 每行一个token（换行分隔）
        2. 逗号分隔的token
        3. 混合格式（同时支持换行和逗号分隔）
        """
        tokens = []
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    if not content:
                        logger.debug(f"📄 Token文件为空: {file_path}")
                        return tokens

                    # 智能解析：同时支持换行和逗号分隔
                    # 1. 先按换行符分割处理每一行
                    lines = content.split('\n')

                    for line in lines:
                        line = line.strip()
                        # 跳过空行和注释行
                        if not line or line.startswith('#'):
                            continue

                        # 2. 检查当前行是否包含逗号分隔
                        if ',' in line:
                            # 按逗号分割当前行
                            comma_tokens = line.split(',')
                            for token in comma_tokens:
                                token = token.strip()
                                if token:  # 跳过空token
                                    tokens.append(token)
                        else:
                            # 整行作为一个token
                            tokens.append(line)

                logger.info(f"📄 从文件加载了 {len(tokens)} 个token: {file_path}")
            else:
                logger.debug(f"📄 Token文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"❌ 读取token文件失败 {file_path}: {e}")
        return tokens

    @property
    def auth_token_list(self) -> List[str]:
        """
        解析认证token列表

        从AUTH_TOKENS_FILE指定的文件加载token（如果配置了文件路径）
        """
        # 如果未配置token文件路径，返回空列表
        if not self.AUTH_TOKENS_FILE:
            logger.debug("📄 未配置AUTH_TOKENS_FILE，跳过token文件加载")
            return []

        # 从文件加载token
        tokens = self._load_tokens_from_file(self.AUTH_TOKENS_FILE)

        # 去重，保持顺序
        if tokens:
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    unique_tokens.append(token)
                    seen.add(token)

            # 记录去重信息
            duplicate_count = len(tokens) - len(unique_tokens)
            if duplicate_count > 0:
                logger.warning(f"⚠️ 检测到 {duplicate_count} 个重复token，已自动去重")

            return unique_tokens

        return []

    @property
    def longcat_token_list(self) -> List[str]:
        """
        解析 LongCat token 列表

        从 LONGCAT_TOKENS_FILE 指定的文件加载 token（如果配置了文件路径）
        """
        # 如果未配置token文件路径，返回空列表
        if not self.LONGCAT_TOKENS_FILE:
            logger.debug("📄 未配置LONGCAT_TOKENS_FILE，跳过LongCat token文件加载")
            return []

        # 从文件加载token
        tokens = self._load_tokens_from_file(self.LONGCAT_TOKENS_FILE)

        # 去重，保持顺序
        if tokens:
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    unique_tokens.append(token)
                    seen.add(token)

            # 记录去重信息
            duplicate_count = len(tokens) - len(unique_tokens)
            if duplicate_count > 0:
                logger.warning(f"⚠️ 检测到 {duplicate_count} 个重复LongCat token，已自动去重")

            return unique_tokens

        return []

    # Model Configuration
    GLM45_MODEL: str = os.getenv("GLM45_MODEL", "GLM-4.5")
    GLM45_THINKING_MODEL: str = os.getenv("GLM45_THINKING_MODEL", "GLM-4.5-Thinking")
    GLM45_SEARCH_MODEL: str = os.getenv("GLM45_SEARCH_MODEL", "GLM-4.5-Search")
    AIR_MODEL: str = os.getenv("AIR_MODEL", "GLM-4.5-Air")
    GLM46_MODEL: str = os.getenv("GLM46_MODEL", "GLM-4.6")
    GLM46_GLM45_THINKING_MODEL: str = os.getenv("GLM46_GLM45_THINKING_MODEL", "GLM-4.6-Thinking")
    GLM46_GLM45_SEARCH_MODEL: str = os.getenv("GLM46_GLM45_SEARCH_MODEL", "GLM-4.6-Search")



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
            # K2Think models
            "MBZUAI-IFM/K2-Think": "k2think",
            # LongCat models
            "LongCat-Flash": "longcat",
            "LongCat": "longcat",
            "LongCat-Search": "longcat",
        }

    # Server Configuration
    LISTEN_PORT: int = int(os.getenv("LISTEN_PORT", "8080"))
    DEBUG_LOGGING: bool = os.getenv("DEBUG_LOGGING", "true").lower() == "true"
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "z-ai2api-server")

    ANONYMOUS_MODE: bool = os.getenv("ANONYMOUS_MODE", "true").lower() == "true"
    TOOL_SUPPORT: bool = os.getenv("TOOL_SUPPORT", "true").lower() == "true"
    SCAN_LIMIT: int = int(os.getenv("SCAN_LIMIT", "200000"))
    SKIP_AUTH_TOKEN: bool = os.getenv("SKIP_AUTH_TOKEN", "false").lower() == "true"
    
    # 是否使用客户端传递的 api_key 作为 Z.AI 认证 token
    USE_CLIENT_TOKEN: bool = os.getenv("USE_CLIENT_TOKEN", "false").lower() == "true"

    # LongCat Configuration
    LONGCAT_PASSPORT_TOKEN: Optional[str] = os.getenv("LONGCAT_PASSPORT_TOKEN")
    LONGCAT_TOKENS_FILE: Optional[str] = os.getenv("LONGCAT_TOKENS_FILE")

    # Z.AI Signature Configuration
    ZAI_SIGNATURE_KEY: str = os.getenv("ZAI_SIGNATURE_KEY", "junjie")
    
    # X-FE-Version Header
    X_FE_VERSION: str = os.getenv("X_FE_VERSION", "prod-fe-1.0.106")


    class Config:
        env_file = ".env"


settings = Settings()
