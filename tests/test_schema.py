"""Tests for Pydantic models from app/schema.py."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError


class TestChatRequest:
    """Test ChatRequest validation."""

    def test_valid_creation(self):
        from app.schema import ChatRequest
        req = ChatRequest(query="Hello, how are you?")
        assert req.query == "Hello, how are you?"
        assert req.conversation_id is None

    def test_with_conversation_id(self):
        from app.schema import ChatRequest
        req = ChatRequest(query="Hi", conversation_id="abc-123_test")
        assert req.conversation_id == "abc-123_test"

    def test_invalid_conversation_id_rejected(self):
        from app.schema import ChatRequest
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="Hi", conversation_id="invalid id with spaces!")
        assert "conversation_id" in str(exc_info.value)

    def test_empty_query_rejected(self):
        from app.schema import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(query="")

    def test_query_max_length(self):
        from app.schema import ChatRequest
        # Just under the limit should work (50K)
        req = ChatRequest(query="a" * 50_000)
        assert len(req.query) == 50_000

        # Over the limit should fail
        with pytest.raises(ValidationError):
            ChatRequest(query="a" * 50_001)


class TestIngestRequest:
    """Test IngestRequest validation."""

    def test_text_only(self):
        from app.schema import IngestRequest
        req = IngestRequest(text="Some document content")
        assert req.text == "Some document content"
        assert req.url is None

    def test_url_only(self):
        from app.schema import IngestRequest
        req = IngestRequest(url="https://example.com/doc")
        assert req.url == "https://example.com/doc"
        assert req.text is None

    def test_both_text_and_url(self):
        from app.schema import IngestRequest
        req = IngestRequest(text="content", url="https://example.com")
        assert req.text == "content"
        assert req.url == "https://example.com"

    def test_neither_text_nor_url_raises(self):
        from app.schema import IngestRequest
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest()
        assert "text" in str(exc_info.value).lower() or "url" in str(exc_info.value).lower()

    def test_invalid_url_scheme(self):
        from app.schema import IngestRequest
        with pytest.raises(ValidationError):
            IngestRequest(url="ftp://example.com/doc")


class TestUserFact:
    """Test UserFact and UserFactCreate models."""

    def test_user_fact_creation(self):
        from app.schema import UserFact
        fact = UserFact(key="favorite_color", value="blue")
        assert fact.key == "favorite_color"
        assert fact.value == "blue"
        assert fact.source == "user_stated"
        assert fact.confidence == 1.0
        assert fact.category == "fact"
        assert fact.id is None

    def test_user_fact_create_validation(self):
        from app.schema import UserFactCreate
        fc = UserFactCreate(key="name", value="Alice")
        assert fc.key == "name"
        assert fc.source == "user_stated"
        assert fc.category == "fact"

    def test_user_fact_create_empty_key_rejected(self):
        from app.schema import UserFactCreate
        with pytest.raises(ValidationError):
            UserFactCreate(key="", value="test")


class TestLessonInfo:
    """Test LessonInfo model."""

    def test_lesson_info_with_lesson_text(self):
        from app.schema import LessonInfo
        lesson = LessonInfo(
            id=1,
            topic="Python",
            wrong_answer="print is a statement",
            correct_answer="print is a function in Python 3",
            lesson_text="User corrected me about print being a function.",
            confidence=0.9,
            times_retrieved=5,
            times_helpful=3,
            created_at="2026-03-19T12:00:00",
        )
        assert lesson.lesson_text == "User corrected me about print being a function."
        assert lesson.id == 1

    def test_lesson_info_lesson_text_optional(self):
        from app.schema import LessonInfo
        lesson = LessonInfo(
            id=2,
            topic="Math",
            wrong_answer=None,
            correct_answer="2+2=4",
            confidence=1.0,
            times_retrieved=0,
            times_helpful=0,
            created_at="2026-03-19",
        )
        assert lesson.lesson_text is None


class TestSkillInfo:
    """Test SkillInfo model."""

    def test_skill_info_with_all_fields(self):
        from app.schema import SkillInfo
        skill = SkillInfo(
            id=1,
            name="weather_check",
            trigger_pattern="weather in (.*)",
            steps=[{"action": "search", "query": "{city} weather"}],
            answer_template="The weather in {city} is {result}.",
            times_used=10,
            success_rate=0.95,
            enabled=True,
            created_at="2026-03-19T10:00:00",
        )
        assert skill.answer_template == "The weather in {city} is {result}."
        assert skill.created_at == "2026-03-19T10:00:00"

    def test_skill_info_optional_fields(self):
        from app.schema import SkillInfo
        skill = SkillInfo(
            id=2,
            name="basic_skill",
            trigger_pattern="do something",
            steps=[],
            times_used=0,
            success_rate=1.0,
            enabled=True,
        )
        assert skill.answer_template is None
        assert skill.created_at is None


class TestStreamEvent:
    """Test StreamEvent.to_sse() produces valid SSE format."""

    def test_to_sse_token_event(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(type=EventType.TOKEN, data={"text": "Hello"})
        sse = event.to_sse()
        assert sse.startswith("event: token\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")
        # Parse the data line
        data_line = sse.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["text"] == "Hello"

    def test_to_sse_done_event(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(type=EventType.DONE, data={"conversation_id": "abc"})
        sse = event.to_sse()
        assert "event: done\n" in sse

    def test_to_sse_empty_data(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(type=EventType.THINKING, data={})
        sse = event.to_sse()
        assert "event: thinking\n" in sse
        assert "data: {}" in sse


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_default_values(self):
        from app.schema import HealthResponse
        hr = HealthResponse()
        assert hr.status == "ok"
        assert hr.timestamp == ""

    def test_custom_values(self):
        from app.schema import HealthResponse
        hr = HealthResponse(
            status="ok",
            timestamp="2026-03-20T12:00:00",
        )
        assert hr.status == "ok"
        assert hr.timestamp == "2026-03-20T12:00:00"


# ===========================================================================
# SSE Parsing and Round-Trip (from test_sse_parsing)
# ===========================================================================

class TestSSEFormatting:
    """Comprehensive SSE formatting tests for all event types."""

    def test_tool_use_event(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(
            type=EventType.TOOL_USE,
            data={"tool": "calculator", "status": "executing", "args": {"expression": "2+2"}},
        )
        sse = event.to_sse()
        assert "event: tool_use\n" in sse
        parsed_data = json.loads(sse.split("data: ")[1].strip())
        assert parsed_data["tool"] == "calculator"

    def test_all_event_types_have_sse(self):
        from app.schema import StreamEvent, EventType
        for event_type in EventType:
            event = StreamEvent(type=event_type, data={"test": True})
            sse = event.to_sse()
            assert f"event: {event_type.value}\n" in sse
            assert "data: " in sse
            assert sse.endswith("\n\n")

    def test_event_type_values(self):
        from app.schema import EventType
        expected = {"thinking", "token", "tool_use", "sources",
                    "lesson_used", "lesson_learned", "warning", "done", "error"}
        actual = {e.value for e in EventType}
        assert expected == actual


class TestSSERoundTrip:
    """Test that SSE output can be correctly round-tripped."""

    def _parse_sse_events(self, sse_stream: str) -> list[dict]:
        events = []
        current_event = None
        current_data = None
        for line in sse_stream.split("\n"):
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                current_data = line[6:]
            elif line == "" and current_event is not None:
                try:
                    parsed = json.loads(current_data) if current_data else {}
                except json.JSONDecodeError:
                    parsed = {"_raw": current_data}
                events.append({"type": current_event, "data": parsed})
                current_event = None
                current_data = None
        return events

    def test_round_trip_multiple_events(self):
        from app.schema import StreamEvent, EventType
        events = [
            StreamEvent(type=EventType.THINKING, data={"status": "started"}),
            StreamEvent(type=EventType.TOKEN, data={"text": "The "}),
            StreamEvent(type=EventType.TOKEN, data={"text": "answer "}),
            StreamEvent(type=EventType.TOKEN, data={"text": "is 42."}),
            StreamEvent(type=EventType.DONE, data={"conversation_id": "c1", "intent": "general"}),
        ]
        sse_stream = "".join(e.to_sse() for e in events)
        parsed = self._parse_sse_events(sse_stream)

        assert len(parsed) == 5
        assert parsed[0]["type"] == "thinking"
        assert parsed[3]["data"]["text"] == "is 42."
        assert parsed[4]["data"]["conversation_id"] == "c1"

    def test_event_ordering_preserved(self):
        from app.schema import StreamEvent, EventType
        events = [
            StreamEvent(type=EventType.THINKING, data={}),
            StreamEvent(type=EventType.TOOL_USE, data={"tool": "web_search", "status": "executing"}),
            StreamEvent(type=EventType.TOOL_USE, data={"tool": "web_search", "status": "complete"}),
            StreamEvent(type=EventType.TOKEN, data={"text": "Result"}),
            StreamEvent(type=EventType.SOURCES, data={"sources": []}),
            StreamEvent(type=EventType.DONE, data={"conversation_id": "c1"}),
        ]
        sse_stream = "".join(e.to_sse() for e in events)
        parsed = self._parse_sse_events(sse_stream)
        types = [p["type"] for p in parsed]
        assert types == ["thinking", "tool_use", "tool_use", "token", "sources", "done"]

    def test_special_characters_roundtrip(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(
            type=EventType.TOKEN,
            data={"text": 'He said "hello" and used a \\ backslash\nNew line'},
        )
        sse = event.to_sse()
        parsed = self._parse_sse_events(sse)
        assert len(parsed) == 1
        assert '"hello"' in parsed[0]["data"]["text"]

    def test_unicode_roundtrip(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(
            type=EventType.TOKEN,
            data={"text": "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059"},
        )
        sse = event.to_sse()
        parsed = self._parse_sse_events(sse)
        assert "\u6771\u4eac" in parsed[0]["data"]["text"]
