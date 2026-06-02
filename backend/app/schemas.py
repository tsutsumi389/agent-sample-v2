from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="記憶を分離するユーザー識別子")
    thread_id: str = Field(..., description="会話履歴を分離するスレッド識別子")
    message: str = Field(..., description="ユーザーの発話")


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class MemoryItem(BaseModel):
    key: str
    value: Any
    created_at: str | None = None
    updated_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str
