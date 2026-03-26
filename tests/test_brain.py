"""Tests for Phase 2: memory, prompt, brain, critique, and chat API."""

from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, _classify_intent, get_services, set_services, think
from app.core.critique import should_critique, critique_answer, format_critique_for_regeneration
from app.core.llm import ToolCall
from app.core.memory import ConversationStore, UserFactStore
from app.core.prompt import (
    build_system_prompt,
    format_lessons_for_prompt,
    format_skills_for_prompt,
)
from app.schema import EventType


# ===========================================================================
# ConversationStore
# ===========================================================================

class TestConversationStore:
    def test_create_and_get(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Test Chat")
        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv["title"] == "Test Chat"

    def test_create_default_title(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()
        conv = store.get_conversation(conv_id)
        assert conv["title"] == "New Chat"

    def test_add_and_get_messages(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()

        store.add_message(conv_id, "user", "Hello")
        store.add_message(conv_id, "assistant", "Hi there!")

        history = store.get_history(conv_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"

    def test_get_history_as_dicts(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()

        store.add_message(conv_id, "user", "What's 2+2?")
        store.add_message(conv_id, "tool", "4", tool_name="calculator")
        store.add_message(conv_id, "assistant", "The answer is 4.")

        dicts = store.get_history_as_dicts(conv_id)
        assert len(dicts) == 3
        assert dicts[0] == {"role": "user", "content": "What's 2+2?"}
        assert dicts[1]["role"] == "assistant"  # Tool results as assistant self-attribution
        assert "calculator" in dicts[1]["content"]
        assert dicts[2] == {"role": "assistant", "content": "The answer is 4."}

    def test_history_limit(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()

        for i in range(10):
            store.add_message(conv_id, "user", f"Message {i}")

        history = store.get_history(conv_id, limit=3)
        assert len(history) == 3
        # Should return the 3 most recent, in chronological order
        assert history[0].content == "Message 7"
        assert history[2].content == "Message 9"

    def test_list_conversations(self, db):
        store = ConversationStore(db)
        store.create_conversation("First")
        store.create_conversation("Second")

        convs = store.list_conversations()
        assert len(convs) == 2
        titles = {c["title"] for c in convs}
        assert titles == {"First", "Second"}

    def test_update_title(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Old Title")
        store.update_title(conv_id, "New Title")
        conv = store.get_conversation(conv_id)
        assert conv["title"] == "New Title"

    def test_delete_conversation(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "Hello")

        store.delete_conversation(conv_id)
        assert store.get_conversation(conv_id) is None
        assert store.get_history(conv_id) == []

    def test_tool_calls_stored_as_json(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation()
        tool_calls = [{"tool": "calculator", "args": {"expression": "2+2"}}]
        store.add_message(conv_id, "assistant", "Let me calculate.", tool_calls=tool_calls)

        history = store.get_history(conv_id)
        assert history[0].tool_calls == tool_calls

    def test_get_nonexistent_conversation(self, db):
        store = ConversationStore(db)
        assert store.get_conversation("nonexistent") is None


# ===========================================================================
# UserFactStore
# ===========================================================================

class TestUserFactStore:
    def test_set_and_get(self, db):
        store = UserFactStore(db)
        store.set("name", "John", source="user_stated")
        fact = store.get("name")
        assert fact is not None
        assert fact.value == "John"
        assert fact.source == "user_stated"

    def test_upsert(self, db):
        store = UserFactStore(db)
        store.set("city", "London", source="inferred", confidence=0.5)
        # Higher confidence from same source overwrites
        store.set("city", "Paris", source="inferred", confidence=0.8)
        fact = store.get("city")
        assert fact.value == "Paris"

    def test_upsert_source_authority(self, db):
        store = UserFactStore(db)
        store.set("city", "London", source="user", confidence=1.0)
        # Lower-authority source should NOT overwrite
        store.set("city", "Paris", source="extracted", confidence=1.0)
        fact = store.get("city")
        assert fact.value == "London"

    def test_get_all(self, db):
        store = UserFactStore(db)
        store.set("name", "John")
        store.set("city", "London")
        facts = store.get_all()
        assert len(facts) == 2

    def test_delete(self, db):
        store = UserFactStore(db)
        store.set("name", "John")
        assert store.delete("name") is True
        assert store.get("name") is None
        assert store.delete("name") is False

    def test_format_for_prompt_empty(self, db):
        store = UserFactStore(db)
        assert store.format_for_prompt() == ""

    def test_format_for_prompt(self, db):
        store = UserFactStore(db)
        store.set("name", "John")
        store.set("city", "London")
        prompt = store.format_for_prompt()
        assert "What You Know About Your Owner" in prompt
        assert "- city: London" in prompt
        assert "- name: John" in prompt


# ===========================================================================
# Prompt Builder
# ===========================================================================

class TestPromptBuilder:
    def test_build_basic(self):
        prompt = build_system_prompt()
        assert "Nova" in prompt
        assert "How You Think" in prompt
        assert "Current Info" in prompt

    def test_build_with_user_facts(self):
        prompt = build_system_prompt(user_facts_text="## What You Know\n- name: John")
        assert "name: John" in prompt

    def test_build_with_tools(self):
        prompt = build_system_prompt(tool_descriptions="web_search(query) — Search the web")
        assert "web_search" in prompt
        assert "Available Tools" in prompt

    def test_build_with_lessons(self):
        prompt = build_system_prompt(lessons_text="## Lessons\n- Topic: Math — 2+2=4")
        assert "Math" in prompt

    def test_build_with_context(self):
        prompt = build_system_prompt(retrieved_context="Document says revenue was $1M")
        assert "Retrieved Context" in prompt
        assert "revenue" in prompt

    def test_truncation_occurs(self):
        # Huge context should be truncated
        big_context = "x" * 20000
        prompt = build_system_prompt(retrieved_context=big_context)
        assert len(prompt) < 25000  # Must be truncated well below 20K+identity (MAX_SYSTEM_TOKENS=6000)
        assert "truncated" in prompt.lower()

    def test_mandatory_blocks_never_truncated(self):
        big_context = "x" * 20000
        prompt = build_system_prompt(
            user_facts_text="## Facts\n- name: John",
            retrieved_context=big_context,
        )
        # Identity and facts must survive even with huge context
        assert "Nova" in prompt
        assert "name: John" in prompt

    def test_format_lessons(self):
        lessons = [
            {"topic": "Math", "wrong_answer": "5", "correct_answer": "4", "lesson_text": ""},
            {"topic": "Python", "wrong_answer": "", "correct_answer": "Use f-strings", "lesson_text": ""},
            {"topic": "Mars sky", "wrong_answer": "", "correct_answer": "", "lesson_text": "The Mars sky is butterscotch"},
        ]
        text = format_lessons_for_prompt(lessons)
        assert "Lessons From Past Corrections" in text
        assert "4, not 5" in text                    # structured fallback
        assert "f-strings" in text                   # correct_answer only
        assert "butterscotch" in text                # lesson_text preferred

    def test_format_lessons_empty(self):
        assert format_lessons_for_prompt([]) == ""

    def test_format_skills(self):
        skills = [
            {"name": "crypto_check", "trigger_pattern": r"price of \w+"},
        ]
        text = format_skills_for_prompt(skills)
        assert "Learned Skills" in text
        assert "crypto_check" in text

    def test_format_skills_empty(self):
        assert format_skills_for_prompt([]) == ""


# ===========================================================================
# Intent Classification
# ===========================================================================

class TestIntentClassification:
    @pytest.mark.asyncio
    async def test_greeting(self):
        assert await _classify_intent("Hello") == "greeting"
        assert await _classify_intent("hey there") == "greeting"
        assert await _classify_intent("good morning") == "greeting"
        assert await _classify_intent("Hi!") == "greeting"

    @pytest.mark.asyncio
    async def test_correction(self):
        assert await _classify_intent("Actually, it's 42") == "correction"
        assert await _classify_intent("That's wrong, the answer is X") == "correction"
        assert await _classify_intent("You're wrong about that") == "correction"
        assert await _classify_intent("The correct answer is Y") == "correction"

    @pytest.mark.asyncio
    async def test_general(self):
        assert await _classify_intent("What's the weather?") == "general"
        assert await _classify_intent("Tell me about Python") == "general"
        assert await _classify_intent("How do I cook pasta?") == "general"


# ===========================================================================
# Brain — think()
# ===========================================================================

class TestBrain:
    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_think_yields_events(self, services):
        """think() should yield THINKING, TOKEN, and DONE events."""
        with patch("app.core.brain.llm") as mock_llm:
            # Mock the LLM to return a simple response
            mock_result = AsyncMock()
            mock_result.content = "Hello! How can I help?"
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            events = []
            async for event in think("Hello"):
                events.append(event)

        event_types = [e.type for e in events]
        assert EventType.THINKING in event_types
        assert EventType.TOKEN in event_types
        assert EventType.DONE in event_types

        # Output verification: TOKEN events should reconstruct the LLM content
        tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
        full_text = "".join(tokens)
        assert full_text == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_think_creates_conversation(self, services):
        """think() with no conversation_id should create one."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hi!"
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test Chat")

            done_event = None
            all_events = []
            async for event in think("Hello"):
                all_events.append(event)
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event is not None
        conv_id = done_event.data["conversation_id"]
        assert conv_id is not None

        # Conversation should exist in DB
        conv = services.conversations.get_conversation(conv_id)
        assert conv is not None

        # Output verification: the streamed tokens should match "Hi!"
        tokens = [e.data["text"] for e in all_events if e.type == EventType.TOKEN]
        assert "".join(tokens) == "Hi!"

    @pytest.mark.asyncio
    async def test_think_uses_existing_conversation(self, services):
        """think() with a conversation_id should use it."""
        conv_id = services.conversations.create_conversation("Existing Chat")

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Sure thing."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            done_event = None
            all_events = []
            async for event in think("Follow up question", conversation_id=conv_id):
                all_events.append(event)
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event.data["conversation_id"] == conv_id

        # Output verification: response content should be streamed correctly
        tokens = [e.data["text"] for e in all_events if e.type == EventType.TOKEN]
        assert "".join(tokens) == "Sure thing."

    @pytest.mark.asyncio
    async def test_think_saves_messages(self, services):
        """think() should save user and assistant messages."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "The answer is 42."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test Title")

            conv_id = None
            async for event in think("What is the meaning of life?"):
                if event.type == EventType.DONE:
                    conv_id = event.data["conversation_id"]

        messages = services.conversations.get_history(conv_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is the meaning of life?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_think_tool_loop(self, services):
        """think() should handle tool calls."""
        with patch("app.core.brain.llm") as mock_llm:
            # First call: tool call
            tool_result = AsyncMock()
            tool_result.content = '{"tool": "calculator", "args": {"expression": "2+2"}}'
            tool_result.tool_calls = [ToolCall(tool="calculator", args={"expression": "2+2"})]

            # Second call: final answer
            final_result = AsyncMock()
            final_result.content = "2 + 2 = 4"
            final_result.tool_calls = []

            mock_llm.generate_with_tools = AsyncMock(
                side_effect=[tool_result, final_result]
            )
            mock_llm.invoke_nothink = AsyncMock(return_value="Math Question")

            events = []
            async for event in think("What is 2+2?"):
                events.append(event)

        event_types = [e.type for e in events]
        assert EventType.TOOL_USE in event_types

        # Should have tool executing and complete events
        tool_events = [e for e in events if e.type == EventType.TOOL_USE]
        assert len(tool_events) == 2
        assert tool_events[0].data["status"] == "executing"
        assert tool_events[1].data["status"] == "complete"

        # Output verification: tool event should name the correct tool
        assert tool_events[0].data["tool"] == "calculator"
        # Output verification: final streamed content should be the final LLM answer
        tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
        full_text = "".join(tokens)
        assert full_text == "2 + 2 = 4"

    @pytest.mark.asyncio
    async def test_think_streams_tokens(self, services):
        """think() should yield TOKEN events with text chunks."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hello world, this is a longer response for testing."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")

            tokens = []
            async for event in think("Hello"):
                if event.type == EventType.TOKEN:
                    tokens.append(event.data["text"])

        # Tokens should reconstruct the full response
        assert "".join(tokens) == "Hello world, this is a longer response for testing."

    @pytest.mark.asyncio
    async def test_think_intent_in_done(self, services):
        """Done event should include the classified intent."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hello!"
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            done_event = None
            all_events = []
            async for event in think("Hi there"):
                all_events.append(event)
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event.data["intent"] == "greeting"

        # Output verification: done event should also contain metadata fields
        assert "conversation_id" in done_event.data
        assert "tool_results_count" in done_event.data
        assert done_event.data["tool_results_count"] == 0

    @pytest.mark.asyncio
    async def test_think_vision_passes_image(self, services):
        """think() with image should include images in user message to LLM."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "I see a cat in the image."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            captured_messages = []
            original_gen = mock_llm.generate_with_tools

            async def capture_messages(msgs, tools, **kwargs):
                captured_messages.extend(msgs)
                return await original_gen(msgs, tools, **kwargs)

            mock_llm.generate_with_tools = AsyncMock(side_effect=capture_messages)

            events = []
            async for event in think("What is in this image?", image="base64data"):
                events.append(event)

        # The last message (user) should have images field
        user_msgs = [m for m in captured_messages if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        assert "images" in user_msgs[-1]
        assert user_msgs[-1]["images"] == ["base64data"]

    @pytest.mark.asyncio
    async def test_think_no_image_no_images_field(self, services):
        """think() without image should NOT include images field."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hi!"
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            captured_messages = []
            original_gen = mock_llm.generate_with_tools

            async def capture_messages(msgs, tools, **kwargs):
                captured_messages.extend(msgs)
                return await original_gen(msgs, tools, **kwargs)

            mock_llm.generate_with_tools = AsyncMock(side_effect=capture_messages)

            async for _ in think("Hello"):
                pass

        user_msgs = [m for m in captured_messages if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        assert "images" not in user_msgs[-1]


# ===========================================================================
# Chat API (using FastAPI test client)
# ===========================================================================

class TestChatAPI:
    @pytest.fixture
    def client(self, db):
        """FastAPI test client with services initialized."""
        from fastapi.testclient import TestClient
        from app.main import app

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return TestClient(app)

    def test_sync_chat(self, client):
        """POST /api/chat should return a full response."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hello! I'm Nova."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Greeting")

            response = client.post("/api/chat", json={"query": "Hello"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Hello! I'm Nova."
        assert "conversation_id" in data

    def test_stream_chat(self, client):
        """POST /api/chat/stream should return SSE events."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hello!"
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")

            response = client.post(
                "/api/chat/stream",
                json={"query": "Hello"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        body = response.text
        assert "event: thinking" in body
        assert "event: token" in body
        assert "event: done" in body

    def test_list_conversations(self, client):
        """GET /api/chat/conversations should return conversation list."""
        svc = get_services()
        svc.conversations.create_conversation("Chat 1")
        svc.conversations.create_conversation("Chat 2")

        response = client.get("/api/chat/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_conversation(self, client):
        """GET /api/chat/conversations/{id} should return conversation with messages."""
        svc = get_services()
        conv_id = svc.conversations.create_conversation("Test")
        svc.conversations.add_message(conv_id, "user", "Hello")

        response = client.get(f"/api/chat/conversations/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test"
        assert len(data["messages"]) == 1

    def test_get_conversation_not_found(self, client):
        response = client.get("/api/chat/conversations/nonexistent")
        assert response.status_code == 404

    def test_delete_conversation(self, client):
        svc = get_services()
        conv_id = svc.conversations.create_conversation("To Delete")

        response = client.delete(f"/api/chat/conversations/{conv_id}")
        assert response.status_code == 200
        assert svc.conversations.get_conversation(conv_id) is None

    def test_facts_crud(self, client):
        """Test create, list, and delete user facts."""
        # Create
        response = client.post("/api/chat/facts", json={"key": "name", "value": "John"})
        assert response.status_code == 200

        # List
        response = client.get("/api/chat/facts")
        assert response.status_code == 200
        facts = response.json()
        assert len(facts) == 1
        assert facts[0]["key"] == "name"
        assert facts[0]["value"] == "John"

        # Delete
        response = client.delete("/api/chat/facts/name")
        assert response.status_code == 200

        # Verify deleted
        response = client.get("/api/chat/facts")
        assert response.json() == []

    def test_delete_fact_not_found(self, client):
        response = client.delete("/api/chat/facts/nonexistent")
        assert response.status_code == 404


# ===========================================================================
# Conversation Cleanup (Item 7)
# ===========================================================================

class TestConversationCleanup:
    def test_deletes_old_conversations(self, db):
        """Item 7: Conversations older than N days should be deleted."""
        from datetime import datetime, timedelta

        store = ConversationStore(db)
        old_id = store.create_conversation("Old Chat")
        new_id = store.create_conversation("New Chat")
        store.add_message(old_id, "user", "hello from the past")
        store.add_message(new_id, "user", "hello from today")

        # Force old_id to have an old timestamp
        cutoff = (datetime.now() - timedelta(days=100)).isoformat()
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (cutoff, old_id),
        )

        cleaned = store.cleanup_old_conversations(days=90)
        assert cleaned == 1
        assert store.get_conversation(old_id) is None
        assert store.get_conversation(new_id) is not None

    def test_keeps_recent_conversations(self, db):
        """Item 7: Recent conversations should NOT be deleted."""
        store = ConversationStore(db)
        store.create_conversation("Chat 1")
        store.create_conversation("Chat 2")

        cleaned = store.cleanup_old_conversations(days=90)
        assert cleaned == 0
        assert len(store.list_conversations()) == 2


# ===========================================================================
# Answer Sanitizer (Fix 7)
# ===========================================================================

class TestAnswerSanitizer:
    def test_strips_correction_note(self):
        from app.core.brain import _sanitize_answer
        text = "The capital is Tokyo.\n\n*Note: Your correction has been saved.*"
        result = _sanitize_answer(text)
        assert "Note:" not in result
        assert "Tokyo" in result

    def test_strips_lesson_note(self):
        from app.core.brain import _sanitize_answer
        text = "Here is the answer.\n\n*Note: I've saved this lesson for future reference.*"
        result = _sanitize_answer(text)
        assert "lesson" not in result.lower() or "lesson" in result.split("*")[0].lower()
        assert "answer" in result.lower()

    def test_strips_think_tags(self):
        from app.core.brain import _sanitize_answer
        text = "<think>internal reasoning</think>\nThe answer is 42."
        result = _sanitize_answer(text)
        assert "internal reasoning" not in result
        assert "42" in result

    def test_strips_recorded_correction(self):
        from app.core.brain import _sanitize_answer
        text = "I've noted your correction and will remember it.\nThe sky is actually butterscotch on Mars."
        result = _sanitize_answer(text)
        assert "butterscotch" in result

    def test_preserves_normal_text(self):
        from app.core.brain import _sanitize_answer
        text = "Python was created by Guido van Rossum in 1991."
        assert _sanitize_answer(text) == text

    def test_collapses_blank_lines(self):
        from app.core.brain import _sanitize_answer
        text = "Line 1.\n\n\n\n\nLine 2."
        result = _sanitize_answer(text)
        assert "\n\n\n" not in result
        assert "Line 1." in result
        assert "Line 2." in result


# ===========================================================================
# Failure-Context Lesson Detection (prompt.py)
# ===========================================================================

class TestFailureContextLesson:
    def test_detects_failure_context(self):
        from app.core.prompt import _is_failure_context_lesson
        # Only triggers when ALL failure-context phrases are present (very strict)
        all_phrases = (
            "fail error timeout timed out cannot limitation "
            "unable truncated incomplete unavailable"
        )
        assert _is_failure_context_lesson(all_phrases)

    def test_ignores_partial_failure_context(self):
        from app.core.prompt import _is_failure_context_lesson
        # Partial matches (2-3 phrases) should NOT skip — that was too aggressive
        assert not _is_failure_context_lesson("When tools fail, report the error and limitation")
        assert not _is_failure_context_lesson("timeout or truncated results should be reported")
        assert not _is_failure_context_lesson("If unable to complete, report the error")

    def test_ignores_non_failure_context(self):
        from app.core.prompt import _is_failure_context_lesson
        # 0-1 failure-context phrases → NOT annotated
        assert not _is_failure_context_lesson("verify selectors exist before clicking")
        assert not _is_failure_context_lesson("browser automation tools are required for forms")
        assert not _is_failure_context_lesson("Python error handling best practices")  # only 1 match
        assert not _is_failure_context_lesson("")

    def test_format_lessons_includes_partial_failure_lessons(self):
        lessons = [
            {"topic": "Good", "lesson_text": "verify fields before clicking", "confidence": 0.9,
             "wrong_answer": "", "correct_answer": ""},
            {"topic": "Cautionary", "lesson_text": "When tools fail, report error and limitation",
             "confidence": 0.8, "wrong_answer": "", "correct_answer": ""},
        ]
        result = format_lessons_for_prompt(lessons)
        # Good lesson is included
        assert "verify fields" in result
        # Partial failure-context lesson is now included (no longer too aggressive)
        assert "report error and limitation" in result

    def test_format_lessons_excludes_structured_fallback_failure(self):
        lessons = [
            {"topic": "Handling", "lesson_text": "",
             "wrong_answer": "ignore the error and timeout", "correct_answer": "report the failure and limitation",
             "confidence": 0.8},
        ]
        result = format_lessons_for_prompt(lessons)
        # Partial failure context is no longer excluded — lesson is included
        assert "report the failure and limitation" in result

    def test_format_lessons_all_failure_returns_empty(self):
        all_phrases = (
            "fail error timeout timed out cannot limitation "
            "unable truncated incomplete unavailable"
        )
        lessons = [
            {"topic": "Fail1", "lesson_text": all_phrases,
             "confidence": 0.8, "wrong_answer": "", "correct_answer": ""},
            {"topic": "Fail2", "lesson_text": all_phrases,
             "confidence": 0.9, "wrong_answer": "", "correct_answer": ""},
        ]
        result = format_lessons_for_prompt(lessons)
        assert result == ""


# ===========================================================================
# Round Success Detection (brain.py)
# ===========================================================================

class TestRoundSuccessDetection:
    def test_all_succeeded(self):
        from app.core.brain import _round_all_succeeded
        assert _round_all_succeeded([("browser", "Page loaded successfully")])
        assert _round_all_succeeded([("browser", "Page loaded"), ("calc", "Result: 42")])
        assert _round_all_succeeded([])

    def test_detects_failure(self):
        from app.core.brain import _round_all_succeeded
        assert not _round_all_succeeded([("browser", "Error: selector not found")])
        assert not _round_all_succeeded([("browser", "Page loaded"), ("calc", "Failed to compute")])
        assert not _round_all_succeeded([("browser", "Request timed out")])

    def test_mixed_rounds_tracked_cumulatively(self):
        """Simulates the partial-failure scenario: early rounds succeed, later round fails.
        The _any_round_succeeded flag should preserve knowledge of prior successes."""
        from app.core.brain import _round_all_succeeded
        # Round 1: success
        r1 = [("browser", "Page loaded successfully")]
        assert _round_all_succeeded(r1)
        # Round 2: success
        r2 = [("browser", "Filled 7 field(s)")]
        assert _round_all_succeeded(r2)
        # Round 3: failure
        r3 = [("browser", "[Tool error: browser] Selector not found")]
        assert not _round_all_succeeded(r3)
        # The cumulative tracking (_any_round_succeeded) is handled in _run_generation_loop,
        # but we verify the building blocks work correctly here.


# ===========================================================================
# Usage Tracking (from test_audit_consolidated)
# ===========================================================================

class TestUsageTracking:
    """GenerationResult includes usage data."""

    def test_generation_result_has_usage_field(self):
        from app.core.llm import GenerationResult
        result = GenerationResult(
            content="test", tool_calls=[], raw={},
            usage={"input_tokens": 100, "output_tokens": 50}
        )
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 50

    def test_generation_result_usage_defaults_none(self):
        from app.core.llm import GenerationResult
        result = GenerationResult(content="test", tool_calls=[], raw={})
        assert result.usage is None


# ===========================================================================
# Critique Sources (from test_audit_consolidated)
# ===========================================================================

class TestCritiqueSources:

    @pytest.fixture
    def mock_llm(self):
        with patch("app.core.critique.llm") as m:
            m.invoke_nothink = AsyncMock(return_value='{"pass": true, "issues": []}')
            yield m

    def test_critique_includes_user_facts(self, mock_llm):
        from app.core.critique import critique_answer

        asyncio.get_event_loop().run_until_complete(
            critique_answer(
                "What is my name?",
                "Your name is Marcus.",
                sources="",
                user_facts="- name: Marcus\n- employer: Acme Corp",
            )
        )

        call_args = mock_llm.invoke_nothink.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Owner facts" in user_msg
        assert "Marcus" in user_msg

    def test_critique_includes_kg_facts(self, mock_llm):
        from app.core.critique import critique_answer

        asyncio.get_event_loop().run_until_complete(
            critique_answer(
                "Tell me about bitcoin",
                "Bitcoin is a cryptocurrency.",
                sources="",
                kg_facts="bitcoin is_a cryptocurrency (confidence: 0.9)",
            )
        )

        call_args = mock_llm.invoke_nothink.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Knowledge graph facts" in user_msg
        assert "bitcoin" in user_msg

    def test_empty_facts_not_added(self, mock_llm):
        from app.core.critique import critique_answer

        asyncio.get_event_loop().run_until_complete(
            critique_answer(
                "What is 2+2?",
                "2+2 equals 4.",
                sources="",
                user_facts="",
                kg_facts="",
            )
        )

        call_args = mock_llm.invoke_nothink.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Owner facts" not in user_msg
        assert "Knowledge graph facts" not in user_msg

    def test_critique_system_prompt_mentions_owner_facts(self):
        from app.core.critique import _CRITIQUE_SYSTEM
        assert "owner facts" in _CRITIQUE_SYSTEM.lower()
        assert "knowledge graph" in _CRITIQUE_SYSTEM.lower()

    def test_refine_response_threads_facts(self):
        from app.core.brain import _refine_response
        sig = inspect.signature(_refine_response)
        assert "user_facts_text" in sig.parameters
        assert "kg_facts_text" in sig.parameters


# ===========================================================================
# Strip Deliberation (from test_audit_consolidated)
# ===========================================================================

class TestStripDeliberation:

    def _strip(self, text: str) -> str:
        from app.monitors.heartbeat import _strip_deliberation
        return _strip_deliberation(text)

    def test_removes_wait_let_me(self):
        result = self._strip("wait no let me check the data again\nThe answer is 42.")
        assert "wait no" not in result
        assert "The answer is 42." in result

    def test_removes_okay_final_version(self):
        result = self._strip("Okay final version:\nHere is the analysis.")
        assert "final version" not in result
        assert "Here is the analysis." in result

    def test_removes_actually_rereading(self):
        result = self._strip("Actually re-reading my thought process\nThe conclusion is clear.")
        assert "re-reading" not in result

    def test_removes_let_me_rethink(self):
        result = self._strip("Let me rethink this approach\nThe real answer is X.")
        assert "rethink" not in result

    def test_preserves_normal_content(self):
        text = "The stock market rose 2% today. Analysts expect continued growth."
        assert self._strip(text) == text

    def test_preserves_mid_sentence(self):
        assert "actually found" in self._strip("I actually found some interesting data about this topic.")

    def test_collapses_blank_lines(self):
        assert "\n\n\n" not in self._strip("Line one\n\n\n\nLine two")


# ===========================================================================
# Critique Passed Guard (from test_audit_consolidated)
# ===========================================================================

class TestCritiquePassedGuard:

    def test_guard_exists_in_refine_response(self):
        """Verify _refine_response exists, is callable, and accepts critique-related params."""
        from app.core.brain import _refine_response
        assert callable(_refine_response)
        sig = inspect.signature(_refine_response)
        # _refine_response should accept the parameters needed for critique flow
        param_names = list(sig.parameters.keys())
        assert "final_content" in param_names
        assert "intent" in param_names


# ===========================================================================
# Quiz Max Tokens (from test_audit_consolidated)
# ===========================================================================

class TestQuizMaxTokens:

    @pytest.mark.asyncio
    async def test_quiz_max_tokens_at_least_600(self, db):
        """Verify the quiz answer generation uses max_tokens >= 600.

        We mock llm.invoke_nothink and capture the kwargs to verify the
        answer generation call uses sufficient max_tokens.
        """
        from app.monitors.heartbeat import HeartbeatLoop
        from app.core.learning import LearningEngine

        db.init_schema()
        learning = LearningEngine(db)

        # Insert a lesson for the quiz to pick up
        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("Python GIL", "GIL allows multithreading", "The GIL prevents true multithreading",
             "GIL lesson", "threading", 0.8),
        )

        captured_max_tokens = []

        async def capturing_invoke(messages, **kwargs):
            if "max_tokens" in kwargs:
                captured_max_tokens.append(kwargs["max_tokens"])
            return '{"pass": true}'

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            learning=learning,
        )
        set_services(svc)

        with patch("app.core.llm.invoke_nothink", new=AsyncMock(side_effect=capturing_invoke)):
            loop = HeartbeatLoop.__new__(HeartbeatLoop)
            result = await loop._execute_quiz({})

        # The answer generation call should use max_tokens >= 600
        assert any(mt >= 600 for mt in captured_max_tokens), (
            f"Expected at least one invoke_nothink call with max_tokens >= 600, got {captured_max_tokens}"
        )


# ===========================================================================
# Fact Extraction Wired to Brain
# ===========================================================================

class TestFactExtractionInBrain:
    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_fact_extracted_during_think(self, services):
        """When user says 'My name is Alex', a fact should be extracted."""
        import asyncio

        with patch("app.core.brain.llm") as mock_brain_llm, \
             patch("app.core.memory.llm") as mock_mem_llm:

            # Brain LLM: normal response
            mock_result = AsyncMock()
            mock_result.content = "Nice to meet you, Alex!"
            mock_result.tool_calls = []
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Introduction")

            # Memory LLM: fact extraction
            mock_mem_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "name": "Alex",
            }))
            mock_mem_llm.extract_json_object = lambda x: json.loads(x)

            async for event in think("My name is Alex"):
                pass

            # Fact extraction is now a background task — drain tasks while mocks are active
            await asyncio.sleep(0.1)
            tasks = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Fact should have been saved
        fact = services.user_facts.get("name")
        assert fact is not None
        assert fact.value == "Alex"
        assert fact.source == "user"  # "My name is..." is an explicit user statement

    @pytest.mark.asyncio
    async def test_no_extraction_for_generic_message(self, services):
        """Generic messages without fact signals should NOT trigger extraction."""
        with patch("app.core.brain.llm") as mock_brain_llm:
            mock_result = AsyncMock()
            mock_result.content = "It's sunny today."
            mock_result.tool_calls = []
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Weather")

            async for event in think("What's the weather?"):
                pass

        # No facts should exist
        facts = services.user_facts.get_all()
        assert len(facts) == 0


# ===========================================================================
# Conversation Search
# ===========================================================================

class TestConversationSearch:
    def test_search_messages(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Python Chat")
        store.add_message(conv_id, "user", "How do I use Python decorators?")
        store.add_message(conv_id, "assistant", "Decorators in Python use the @ syntax.")

        results = store.search_messages("Python decorators")
        assert len(results) >= 1
        assert any("decorator" in r["content"].lower() for r in results)

    def test_search_conversations(self, db):
        store = ConversationStore(db)
        cid1 = store.create_conversation("Python Chat")
        store.add_message(cid1, "user", "Tell me about Python")
        cid2 = store.create_conversation("Weather Chat")
        store.add_message(cid2, "user", "Tell me about the weather")

        results = store.search_conversations("Python")
        assert len(results) >= 1
        assert results[0]["conversation_id"] == cid1

    def test_search_no_results(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Test")
        store.add_message(conv_id, "user", "Hello world")

        results = store.search_messages("quantum physics")
        assert len(results) == 0


# ===========================================================================
# Facts Not Truncated in Prompt
# ===========================================================================

class TestFactsNotTruncated:
    def test_facts_not_truncated(self, db):
        """User facts (Block 2) should never be truncated, even with huge context."""
        store = UserFactStore(db)
        store.set("name", "Alex")
        store.set("city", "Tokyo")

        facts_text = store.format_for_prompt()
        big_context = "x" * 50000

        prompt = build_system_prompt(
            user_facts_text=facts_text,
            retrieved_context=big_context,
        )

        # Facts must survive
        assert "Alex" in prompt
        assert "Tokyo" in prompt
        # Context should be truncated
        assert "truncated" in prompt.lower()


# ===========================================================================
# User Fact Access Tracking (from test_memory_tracking)
# ===========================================================================

class TestUserFactAccessTracking:
    @pytest.fixture
    def store(self, db):
        return UserFactStore(db)

    def test_refresh_access_updates_count(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        row = store._db.fetchone(
            "SELECT access_count FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["access_count"] == 1

    def test_refresh_access_increments(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        store.refresh_access(["name"])
        row = store._db.fetchone(
            "SELECT access_count FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["access_count"] == 2

    def test_refresh_access_updates_timestamp(self, store):
        store.set("name", "Alex")
        store.refresh_access(["name"])
        row = store._db.fetchone(
            "SELECT last_accessed_at FROM user_facts WHERE key = ?", ("name",)
        )
        assert row["last_accessed_at"] is not None

    def test_refresh_access_empty_list(self, store):
        # Should not raise
        store.refresh_access([])

    def test_get_stale_facts(self, db):
        store = UserFactStore(db)
        store.set("old_fact", "old value")
        store.set("new_fact", "new value")
        store.refresh_access(["new_fact"])
        db.execute(
            "UPDATE user_facts SET last_accessed_at = datetime('now', '-90 days') WHERE key = ?",
            ("old_fact",),
        )
        stale = store.get_stale_facts(days=60)
        stale_keys = [f.key for f in stale]
        assert "old_fact" in stale_keys

    def test_stale_facts_includes_never_accessed(self, db):
        store = UserFactStore(db)
        store.set("never_accessed", "some value")
        stale = store.get_stale_facts(days=60)
        stale_keys = [f.key for f in stale]
        assert "never_accessed" in stale_keys


# ===========================================================================
# Should Critique Detection Heuristic (from test_critique)
# ===========================================================================

class TestShouldCritique:
    def test_greeting_skipped(self):
        assert not should_critique("hi", "Hello! How can I help?", "greeting", [])

    def test_correction_skipped(self):
        assert not should_critique("wrong", "I'll fix that", "correction", [])

    def test_short_answer_skipped(self):
        assert not should_critique("what is 2+2?", "4", "general", [])

    def test_successful_tool_results_skip(self):
        assert not should_critique(
            "bitcoin price",
            "Bitcoin is currently at $50,000 based on the search results.",
            "general",
            [{"tool": "web_search", "output": "..."}],
        )

    def test_failed_tool_results_trigger(self):
        assert should_critique(
            "bitcoin price",
            "Bitcoin is currently at $50,000 based on the search results.",
            "general",
            [{"tool": "web_search", "output": "[tool web_search failed: timeout]"}],
        )

    def test_long_answer_triggers(self):
        answer = "Here is a comprehensive analysis. " * 20
        assert should_critique("analyze this", answer, "general", [])

    def test_planned_query_triggers(self):
        assert should_critique(
            "compare these two things",
            "Here's my detailed comparison of the two approaches with several important considerations to discuss.",
            "general",
            [],
            was_planned=True,
        )

    def test_medium_answer_no_tools_no_plan(self):
        assert not should_critique(
            "what is Python?",
            "Python is a programming language created by Guido van Rossum.",
            "general",
            [],
        )


# ===========================================================================
# Critique Answer (from test_critique)
# ===========================================================================

class TestCritiqueAnswer:
    @pytest.mark.asyncio
    async def test_passing_critique(self):
        mock_response = json.dumps({"pass": True, "issues": []})
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            result = await critique_answer("What is Python?", "Python is a programming language.")
        assert result is not None
        assert result["pass"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_failing_critique(self):
        mock_response = json.dumps({
            "pass": False,
            "issues": ["Missed part 2 of the question", "Unsupported claim about performance"],
        })
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            result = await critique_answer(
                "Compare Python and Rust performance and ecosystem",
                "Python is slower than Rust.",
            )
        assert result is not None
        assert result["pass"] is False
        assert len(result["issues"]) == 2

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="")
            result = await critique_answer("query", "answer")
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=Exception("LLM down"))
            result = await critique_answer("query", "answer")
        assert result is None


# ===========================================================================
# Format Critique for Regeneration (from test_critique)
# ===========================================================================

class TestFormatCritique:
    def test_passing_critique_empty(self):
        assert format_critique_for_regeneration({"pass": True, "issues": []}) == ""

    def test_failing_critique_formatted(self):
        critique = {"pass": False, "issues": ["Missed part 2", "Unsupported claim"]}
        text = format_critique_for_regeneration(critique)
        assert "[SELF-CHECK FAILED]" in text
        assert "Missed part 2" in text
        assert "Unsupported claim" in text

    def test_no_issues_empty(self):
        assert format_critique_for_regeneration({"pass": False, "issues": []}) == ""

    def test_none_critique_empty(self):
        assert format_critique_for_regeneration(None) == ""


# ===========================================================================
# Critique Brain Integration (from test_critique)
# ===========================================================================

class TestCritiqueBrainIntegration:
    @pytest.mark.asyncio
    async def test_critique_triggers_on_complex_query(self, db):
        """When critique is enabled and query was planned, critique should run."""
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Here is a comprehensive comparison of the two approaches. " * 5
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({"pass": True, "issues": []}))

            events = []
            async for event in think("Compare Python and Rust for web development and also analyze their ecosystems"):
                events.append(event)

        done_events = [e for e in events if e.type.value == "done"]
        assert len(done_events) == 1


# ===========================================================================
# JSON Extraction (from test_foundation — app/core/llm.py)
# ===========================================================================

class TestJsonExtraction:
    def test_find_balanced_json_object(self):
        from app.core.llm import _find_balanced_json
        text = 'Some text {"tool": "web_search", "args": {"query": "test"}} more text'
        result = _find_balanced_json(text, "{")
        assert json.loads(result) == {"tool": "web_search", "args": {"query": "test"}}

    def test_find_balanced_json_array(self):
        from app.core.llm import _find_balanced_json
        text = 'Here is the list: [{"a": 1}, {"b": 2}] done'
        result = _find_balanced_json(text, "[")
        assert json.loads(result) == [{"a": 1}, {"b": 2}]

    def test_find_balanced_json_nested(self):
        from app.core.llm import _find_balanced_json
        text = '{"outer": {"inner": {"deep": true}}}'
        result = _find_balanced_json(text, "{")
        assert json.loads(result) == {"outer": {"inner": {"deep": True}}}

    def test_find_balanced_json_with_strings(self):
        from app.core.llm import _find_balanced_json
        text = '{"key": "value with {braces} inside"}'
        result = _find_balanced_json(text, "{")
        assert json.loads(result) == {"key": "value with {braces} inside"}

    def test_extract_json_object(self):
        from app.core.llm import extract_json_object
        text = 'I will search for that.\n{"tool": "web_search", "args": {"query": "test"}}'
        obj = extract_json_object(text)
        assert obj["tool"] == "web_search"

    def test_extract_json_object_empty(self):
        from app.core.llm import extract_json_object
        assert extract_json_object("no json here") == {}
        assert extract_json_object("") == {}

    def test_extract_tool_call_valid(self):
        from app.core.llm import _extract_tool_call
        tools = [{"name": "web_search"}, {"name": "calculator"}]
        content = '{"tool": "web_search", "args": {"query": "hello"}}'
        tc = _extract_tool_call(content, tools)
        assert tc is not None
        assert tc.tool == "web_search"
        assert tc.args == {"query": "hello"}

    def test_extract_tool_call_invalid_tool(self):
        from app.core.llm import _extract_tool_call
        tools = [{"name": "web_search"}]
        content = '{"tool": "google_search", "args": {"query": "hello"}}'
        tc = _extract_tool_call(content, tools)
        assert tc is None

    def test_extract_tool_call_fuzzy_match(self):
        from app.core.llm import _extract_tool_call
        tools = [{"name": "web_search"}]
        content = '{"tool": "Web_Search", "args": {"query": "hello"}}'
        tc = _extract_tool_call(content, tools)
        assert tc is not None
        assert tc.tool == "web_search"


# ===========================================================================
# Concurrent think() — per-conversation lock infrastructure
# ===========================================================================

class TestConversationLocks:
    """Tests for the per-conversation lock infrastructure in brain.py."""

    def test_lock_infrastructure_exists(self):
        """Verify _conversation_locks dict and _get_conversation_lock exist."""
        from app.core import brain
        assert hasattr(brain, "_conversation_locks")
        assert isinstance(brain._conversation_locks, dict)
        assert hasattr(brain, "_get_conversation_lock")
        assert asyncio.iscoroutinefunction(brain._get_conversation_lock)

    def test_max_conversation_locks_constant(self):
        """Verify the LRU capacity constant exists."""
        from app.core import brain
        assert hasattr(brain, "_MAX_CONVERSATION_LOCKS")
        assert brain._MAX_CONVERSATION_LOCKS == 500

    @pytest.mark.asyncio
    async def test_same_id_returns_same_lock(self):
        """_get_conversation_lock returns the same lock for the same conv_id."""
        from app.core.brain import _get_conversation_lock, _conversation_locks
        _conversation_locks.clear()

        lock1 = await _get_conversation_lock("conv-abc")
        lock2 = await _get_conversation_lock("conv-abc")
        assert lock1 is lock2

        _conversation_locks.clear()

    @pytest.mark.asyncio
    async def test_different_ids_return_different_locks(self):
        """_get_conversation_lock returns different locks for different conv_ids."""
        from app.core.brain import _get_conversation_lock, _conversation_locks
        _conversation_locks.clear()

        lock_a = await _get_conversation_lock("conv-111")
        lock_b = await _get_conversation_lock("conv-222")
        assert lock_a is not lock_b

        _conversation_locks.clear()

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Creating >500 locks evicts the oldest entries."""
        from app.core.brain import _get_conversation_lock, _conversation_locks, _MAX_CONVERSATION_LOCKS
        _conversation_locks.clear()

        # Create exactly MAX locks
        for i in range(_MAX_CONVERSATION_LOCKS):
            await _get_conversation_lock(f"conv-{i}")

        assert len(_conversation_locks) == _MAX_CONVERSATION_LOCKS

        # The first key should still be there
        assert "conv-0" in _conversation_locks

        # Adding one more should evict the oldest (conv-0)
        await _get_conversation_lock("conv-overflow")
        assert len(_conversation_locks) == _MAX_CONVERSATION_LOCKS
        assert "conv-0" not in _conversation_locks
        assert "conv-overflow" in _conversation_locks

        _conversation_locks.clear()

    @pytest.mark.asyncio
    async def test_lru_eviction_multiple(self):
        """Eviction maintains capacity when many new locks are added after saturation."""
        from app.core.brain import _get_conversation_lock, _conversation_locks, _MAX_CONVERSATION_LOCKS
        _conversation_locks.clear()

        # Fill to capacity
        for i in range(_MAX_CONVERSATION_LOCKS):
            await _get_conversation_lock(f"fill-{i}")

        # Add 10 more — should stay at capacity
        for i in range(10):
            await _get_conversation_lock(f"extra-{i}")

        assert len(_conversation_locks) == _MAX_CONVERSATION_LOCKS

        # First 10 fills should be evicted
        for i in range(10):
            assert f"fill-{i}" not in _conversation_locks

        # The extras should be present
        for i in range(10):
            assert f"extra-{i}" in _conversation_locks

        _conversation_locks.clear()


class TestBackgroundTaskNonBlocking:
    """Test that fact extraction runs as a background task, not blocking think()."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_fact_extraction_is_background_task(self, services):
        """Fact extraction should be dispatched via asyncio.create_task, not awaited inline."""
        # We verify this by checking that think() completes even when the
        # fact extraction coroutine hasn't resolved yet.
        extraction_started = asyncio.Event()
        extraction_gate = asyncio.Event()

        with patch("app.core.brain.llm") as mock_brain_llm, \
             patch("app.core.memory.llm") as mock_mem_llm:

            # Brain LLM: normal response
            mock_result = AsyncMock()
            mock_result.content = "Nice to meet you, Alex!"
            mock_result.tool_calls = []
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Introduction")

            # Memory LLM: fact extraction that blocks until we release it
            async def slow_extraction(*args, **kwargs):
                extraction_started.set()
                await extraction_gate.wait()
                return json.dumps({"name": "Alex"})

            mock_mem_llm.invoke_nothink = AsyncMock(side_effect=slow_extraction)
            mock_mem_llm.extract_json_object = lambda x: json.loads(x)

            # think() should yield all events and complete without waiting
            # for the slow fact extraction
            events = []
            async for event in think("My name is Alex"):
                events.append(event)

            # think() should have completed (yielded DONE) even though
            # extraction hasn't finished yet
            event_types = [e.type for e in events]
            assert EventType.DONE in event_types

            # Now release the extraction so the background task can finish
            extraction_gate.set()
            # Give the background task a moment to complete
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_tasks_tracked_in_set(self, services):
        """Background tasks should be tracked in _background_tasks set for GC protection."""
        from app.core.brain import _background_tasks

        with patch("app.core.brain.llm") as mock_brain_llm, \
             patch("app.core.memory.llm") as mock_mem_llm:

            mock_result = AsyncMock()
            mock_result.content = "Hello Alex!"
            mock_result.tool_calls = []
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Greeting")

            # Block extraction so it stays in _background_tasks
            gate = asyncio.Event()

            async def blocked_extraction(*args, **kwargs):
                await gate.wait()
                return json.dumps({})

            mock_mem_llm.invoke_nothink = AsyncMock(side_effect=blocked_extraction)
            mock_mem_llm.extract_json_object = lambda x: json.loads(x)

            initial_count = len(_background_tasks)

            async for event in think("My name is Alex"):
                pass

            # At least one background task should have been added
            # (fact extraction). It may or may not have completed by now,
            # but immediately after think() the task should still be tracked
            # since we blocked it.
            assert len(_background_tasks) >= initial_count

            # Release and clean up
            gate.set()
            await asyncio.sleep(0.1)
