"""Pydantic models for API request/response contracts and SSE events."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = Field(None, max_length=100)
    image_base64: str | None = Field(None, max_length=10_000_000)

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Allow UUIDs, uuid-prefixed IDs, and simple alphanumeric IDs
        if not re.match(r"^[a-zA-Z0-9_\-]{1,100}$", v):
            raise ValueError("conversation_id must be alphanumeric with hyphens/underscores, max 100 chars")
        return v


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    lessons_used: int = 0
    skill_used: str | None = None


# ---------------------------------------------------------------------------
# SSE Events
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    THINKING = "thinking"
    TOKEN = "token"
    TOOL_USE = "tool_use"
    SOURCES = "sources"
    LESSON_USED = "lesson_used"
    LESSON_LEARNED = "lesson_learned"
    WARNING = "warning"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    type: EventType
    data: dict = Field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        import json
        return f"event: {self.type.value}\ndata: {json.dumps(self.data)}\n\n"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class UserFact(BaseModel):
    id: int | None = None
    key: str
    value: str
    source: str = "user_stated"
    confidence: float = 1.0
    category: str = "fact"


class UserFactCreate(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=5_000)
    source: str = Field("user_stated", max_length=100)
    category: str = Field("fact", max_length=50)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    text: str | None = Field(None, max_length=1_000_000)
    url: str | None = Field(None, max_length=2_000)
    title: str | None = Field(None, max_length=500)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v



# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

class LearningMetrics(BaseModel):
    total_lessons: int = 0
    total_skills: int = 0
    total_corrections: int = 0
    training_examples: int = 0
    last_correction_date: str | None = None


class LessonInfo(BaseModel):
    id: int
    topic: str
    wrong_answer: str | None
    correct_answer: str
    confidence: float
    times_retrieved: int
    times_helpful: int
    created_at: str


class SkillInfo(BaseModel):
    id: int
    name: str
    trigger_pattern: str
    steps: list[dict]
    times_used: int
    success_rate: float
    enabled: bool


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model: str = ""
    provider: str = "ollama"
    llm_connected: bool = False
    ollama_connected: bool = False  # backward compat
    db_connected: bool = False


class StatusResponse(BaseModel):
    conversations: int = 0
    messages: int = 0
    user_facts: int = 0
    lessons: int = 0
    skills: int = 0
    documents: int = 0
    training_examples: int = 0
    kg_facts: int = 0
    reflexions: int = 0
    custom_tools: int = 0
