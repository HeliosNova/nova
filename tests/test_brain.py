"""Tests for Phase 2: memory, prompt, brain, and chat API."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.brain import Services, _classify_intent, get_services, set_services, think
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
        assert dicts[1]["role"] == "system"  # Tool results become system messages
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
        store.set("city", "London")
        store.set("city", "Paris")
        fact = store.get("city")
        assert fact.value == "Paris"

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
        assert len(prompt) < 17000  # Must be truncated well below 20K
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
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            events = []
            async for event in think("Hello"):
                events.append(event)

        event_types = [e.type for e in events]
        assert EventType.THINKING in event_types
        assert EventType.TOKEN in event_types
        assert EventType.DONE in event_types

    @pytest.mark.asyncio
    async def test_think_creates_conversation(self, services):
        """think() with no conversation_id should create one."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hi!"
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test Chat")

            done_event = None
            async for event in think("Hello"):
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event is not None
        conv_id = done_event.data["conversation_id"]
        assert conv_id is not None

        # Conversation should exist in DB
        conv = services.conversations.get_conversation(conv_id)
        assert conv is not None

    @pytest.mark.asyncio
    async def test_think_uses_existing_conversation(self, services):
        """think() with a conversation_id should use it."""
        conv_id = services.conversations.create_conversation("Existing Chat")

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Sure thing."
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            done_event = None
            async for event in think("Follow up question", conversation_id=conv_id):
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event.data["conversation_id"] == conv_id

    @pytest.mark.asyncio
    async def test_think_saves_messages(self, services):
        """think() should save user and assistant messages."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "The answer is 42."
            mock_result.tool_call = None
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
            tool_call_obj = AsyncMock()
            tool_call_obj.tool = "calculator"
            tool_call_obj.args = {"expression": "2+2"}
            tool_result.tool_call = tool_call_obj

            # Second call: final answer
            final_result = AsyncMock()
            final_result.content = "2 + 2 = 4"
            final_result.tool_call = None

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

    @pytest.mark.asyncio
    async def test_think_streams_tokens(self, services):
        """think() should yield TOKEN events with text chunks."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Hello world, this is a longer response for testing."
            mock_result.tool_call = None
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
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            done_event = None
            async for event in think("Hi there"):
                if event.type == EventType.DONE:
                    done_event = event

        assert done_event.data["intent"] == "greeting"

    @pytest.mark.asyncio
    async def test_think_vision_passes_image(self, services):
        """think() with image should include images in user message to LLM."""
        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "I see a cat in the image."
            mock_result.tool_call = None
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
            mock_result.tool_call = None
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
            mock_result.tool_call = None
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
            mock_result.tool_call = None
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
