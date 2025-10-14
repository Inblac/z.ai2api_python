#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel


class ContentPart(BaseModel):
    """Content part model for OpenAI's new content format"""

    type: str
    text: Optional[str] = None


class Message(BaseModel):
    """Chat message model"""

    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAIRequest(BaseModel):
    """OpenAI-compatible request model"""

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    thinking: Optional[Dict[str, Any]] = None


class Delta(BaseModel):
    """Stream delta model"""

    role: Optional[str] = None
    content: Optional[str] = "" or None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Choice(BaseModel):
    """Response choice model"""

    index: int
    message: Optional[Message] = None
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage statistics"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIResponse(BaseModel):
    """OpenAI-compatible response model"""

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class Model(BaseModel):
    """Model information for listing"""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Models list response model"""

    object: str = "list"
    data: List[Model]
