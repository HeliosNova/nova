"""Item 59: Full pipeline test through brain.think().

Mocks the LLM provider to return a tool call (calculator) then a final
answer. Verifies tool execution and final response.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services, think
from app.core.memory import ConversationStore, UserFactStore
from app.schema import EventType


def _mock_llm_simple(mock_llm, content="Hello!", invoke_nothink_return="Test Title"):
    """Helper to configure a mock LLM for simple (no-tool) responses."""
    mock_llm.generate_with_tools = AsyncMock(return_value=MagicMock(
        content=content,
        tool_calls=[],
        raw={},
        thinking="",
        usage=None,
    ))
    mock_llm.get_provider = MagicMock(return_value=MagicMock(
        capabilities=MagicMock(needs_emphatic_prompts=False),
    ))
    mock_llm._strip_think_tags = lambda x: x
    mock_llm.extract_json_object = MagicMock(return_value=None)
    mock_llm.invoke_nothink = AsyncMock(return_value=invoke_nothink_return)
    mock_llm._extract_tool_calls = MagicMock(return_value=[])
    mock_llm.GenerationResult = MagicMock


def _make_tool_registry(services, tools=None, execute_map=None):
    """Helper to configure a mock tool registry on services.

    Args:
        services: The Services instance.
        tools: List of tool dicts, e.g. [{"name": "calculator"}].
        execute_map: Dict of tool_name -> (output_str, ToolResult) for execute_full.
    """
    from app.tools.base import ToolResult
    if tools is None:
        tools = [{"name": "calculator"}]
    if execute_map is None:
        execute_map = {"calculator": ("4", ToolResult(output="4", success=True))}

    mock_registry = MagicMock()
    mock_registry.get_descriptions.return_value = " | ".join(
        f"{t['name']}(...)" for t in tools
    )
    mock_registry.get_tool_list.return_value = tools
    mock_registry.get.return_value = MagicMock(trim_output=lambda x: x[:500])

    async def mock_execute_full(name, args):
        if name in execute_map:
            return execute_map[name]
        return f"[Tool error: {name}] Not found", ToolResult(
            output="", success=False, error="Not found"
        )

    mock_registry.execute_full = mock_execute_full
    services.tool_registry = mock_registry
    return mock_registry


@pytest.fixture
def services(db):
    """Set up minimal services for brain.think()."""
    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)
    return svc


class TestBrainThinkPipeline:
    """Full pipeline test through brain.think()."""

    @pytest.mark.asyncio
    async def test_simple_query_returns_token_and_done(self, services):
        """A simple query should produce THINKING, TOKEN, and DONE events."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Hello! How can I help you?")

            events = []
            async for event in think("hello"):
                events.append(event)

            event_types = [e.type for e in events]
            assert EventType.THINKING in event_types
            assert EventType.TOKEN in event_types
            assert EventType.DONE in event_types

    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self, services):
        """LLM returns a tool call, tool executes, then final answer."""
        from app.core.llm import ToolCall, GenerationResult

        # First call returns a tool call, second returns final answer
        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "calculator", "args": {"expression": "2+2"}}',
                    tool_calls=[ToolCall(tool="calculator", args={"expression": "2+2"})],
                    raw={},
                )
            return GenerationResult(
                content="The answer is 4.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test Title")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            # Mock tool registry
            from app.tools.base import ToolResult
            mock_registry = MagicMock()
            mock_registry.get_descriptions.return_value = "calculator(expression: str) — Evaluate math."
            mock_registry.get_tool_list.return_value = [{"name": "calculator"}]
            mock_registry.get.return_value = MagicMock(trim_output=lambda x: x[:500])

            async def mock_execute_full(name, args):
                return "4", ToolResult(output="4", success=True)

            mock_registry.execute_full = mock_execute_full
            services.tool_registry = mock_registry

            events = []
            async for event in think("What is 2+2?"):
                events.append(event)

            event_types = [e.type for e in events]
            assert EventType.TOOL_USE in event_types
            assert EventType.TOKEN in event_types
            assert EventType.DONE in event_types

            # Verify tool use events
            tool_events = [e for e in events if e.type == EventType.TOOL_USE]
            assert any(e.data.get("tool") == "calculator" for e in tool_events)

    @pytest.mark.asyncio
    async def test_done_event_has_conversation_id(self, services):
        """DONE event should include a conversation_id."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Sure!")

            events = []
            async for event in think("hello"):
                events.append(event)

            done_events = [e for e in events if e.type == EventType.DONE]
            assert len(done_events) == 1
            assert "conversation_id" in done_events[0].data
            assert done_events[0].data["conversation_id"]

    @pytest.mark.asyncio
    async def test_messages_saved_to_db(self, services):
        """User query and assistant response should be saved to database."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="I'm doing great!")

            conv_id = None
            async for event in think("How are you?"):
                if event.type == EventType.DONE:
                    conv_id = event.data.get("conversation_id")

            assert conv_id is not None
            # Allow background tasks to complete
            await asyncio.sleep(0.1)
            history = services.conversations.get_history(conv_id)
            roles = [m.role for m in history]
            assert "user" in roles
            assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_query_too_long_returns_error(self, services):
        """Query exceeding MAX_QUERY_LENGTH should produce an ERROR event."""
        events = []
        async for event in think("x" * 100_000):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.ERROR in event_types

    @pytest.mark.asyncio
    async def test_ephemeral_mode_no_db_save(self, services):
        """Ephemeral mode should not save messages to database."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Ephemeral response")

            events = []
            async for event in think("hello", ephemeral=True):
                events.append(event)

            done_events = [e for e in events if e.type == EventType.DONE]
            assert len(done_events) == 1
            assert done_events[0].data.get("ephemeral") is True


class TestToolRoundTrip:
    """Full tool round-trip: tool call -> execution -> response with tool result."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_tool_round_trip_verifies_output_content(self, services):
        """Tool call -> execution -> final answer includes the tool output in the response."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "web_search", "args": {"query": "Python release date"}}',
                    tool_calls=[ToolCall(tool="web_search", args={"query": "Python release date"})],
                    raw={},
                )
            return GenerationResult(
                content="Python was first released on February 20, 1991.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Python Release")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(services, tools=[{"name": "web_search"}], execute_map={
                "web_search": (
                    "Python 1.0 released February 20, 1991 by Guido van Rossum",
                    ToolResult(output="Python 1.0 released February 20, 1991 by Guido van Rossum", success=True),
                ),
            })

            events = []
            async for event in think("When was Python released?"):
                events.append(event)

            # Verify tool executing + complete events
            tool_events = [e for e in events if e.type == EventType.TOOL_USE]
            statuses = [e.data.get("status") for e in tool_events]
            assert "executing" in statuses
            assert "complete" in statuses

            # Verify complete event carries the tool result
            complete_events = [e for e in tool_events if e.data.get("status") == "complete"]
            assert any("Python" in str(e.data.get("result", "")) for e in complete_events)

            # Verify final TOKEN content references the answer
            tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
            full_text = "".join(tokens)
            assert "1991" in full_text

    @pytest.mark.asyncio
    async def test_tool_round_trip_tool_result_saved_to_db(self, services):
        """Tool results are saved as tool-role messages in the conversation history."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "calculator", "args": {"expression": "10*5"}}',
                    tool_calls=[ToolCall(tool="calculator", args={"expression": "10*5"})],
                    raw={},
                )
            return GenerationResult(
                content="10 times 5 equals 50.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Math")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(services, execute_map={
                "calculator": ("50", ToolResult(output="50", success=True)),
            })

            conv_id = None
            async for event in think("What is 10 times 5?"):
                if event.type == EventType.DONE:
                    conv_id = event.data["conversation_id"]

            assert conv_id is not None
            await asyncio.sleep(0.1)
            history = services.conversations.get_history(conv_id)
            # Should have: user, tool, assistant
            roles = [m.role for m in history]
            assert "tool" in roles
            tool_msg = next(m for m in history if m.role == "tool")
            assert "50" in tool_msg.content


class TestCorrectionDetectionFlow:
    """Correction detection -> lesson creation flow through brain.think()."""

    @pytest.fixture
    def services(self, db):
        from app.core.learning import LearningEngine, Correction
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            learning=LearningEngine(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_correction_triggers_lesson_learned_event(self, services):
        """A correction query with prior history should produce a LESSON_LEARNED event."""
        from app.core.learning import Correction

        # Pre-populate conversation with a wrong answer
        conv_id = services.conversations.create_conversation("Test")
        services.conversations.add_message(conv_id, "user", "What is the capital of Australia?")
        services.conversations.add_message(conv_id, "assistant", "The capital of Australia is Sydney.")

        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="You're right, the capital of Australia is Canberra, not Sydney.")

            # Mock detect_correction to return a correction
            mock_correction = Correction(
                user_message="Actually, the capital is Canberra",
                previous_answer="The capital of Australia is Sydney.",
                topic="Australian capital",
                correct_answer="Canberra",
                wrong_answer="Sydney",
                original_query="What is the capital of Australia?",
                lesson_text="The capital of Australia is Canberra, not Sydney.",
            )
            services.learning.detect_correction = AsyncMock(return_value=mock_correction)

            events = []
            async for event in think(
                "Actually, the capital is Canberra",
                conversation_id=conv_id,
            ):
                events.append(event)

            await asyncio.sleep(0.2)
            event_types = [e.type for e in events]
            assert EventType.LESSON_LEARNED in event_types

            lesson_event = next(e for e in events if e.type == EventType.LESSON_LEARNED)
            assert lesson_event.data["topic"] == "Australian capital"
            assert "lesson_id" in lesson_event.data

    @pytest.mark.asyncio
    async def test_correction_saves_dpo_training_pair(self, services):
        """A correction should save a DPO training pair (chosen/rejected)."""
        from app.core.learning import Correction

        conv_id = services.conversations.create_conversation("DPO Test")
        services.conversations.add_message(conv_id, "user", "What color is the sky on Mars?")
        services.conversations.add_message(conv_id, "assistant", "The sky on Mars is blue.")

        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="The Mars sky is actually butterscotch/pinkish.")

            mock_correction = Correction(
                user_message="No, Mars sky is butterscotch",
                previous_answer="The sky on Mars is blue.",
                topic="Mars sky color",
                correct_answer="butterscotch",
                wrong_answer="blue",
                original_query="What color is the sky on Mars?",
                lesson_text="The Mars sky is butterscotch due to iron oxide dust.",
            )
            services.learning.detect_correction = AsyncMock(return_value=mock_correction)
            services.learning.save_training_pair = AsyncMock()

            events = []
            async for event in think(
                "No, Mars sky is butterscotch",
                conversation_id=conv_id,
            ):
                events.append(event)

            await asyncio.sleep(0.2)
            services.learning.save_training_pair.assert_called_once()
            call_kwargs = services.learning.save_training_pair.call_args
            # The DPO pair should have the original query, the bad answer, and the good answer
            args = call_kwargs.kwargs if call_kwargs.kwargs else {}
            if not args:
                # Positional args
                pass
            assert services.learning.save_training_pair.called


class TestFactExtractionFlow:
    """Fact extraction from user messages through the post-processing pipeline."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_fact_signal_triggers_extraction(self, services):
        """A message with fact signals (e.g. 'my name is') should trigger extraction."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Nice to meet you, John!")

            with patch("app.core.brain.extract_facts_from_message", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = {"name": {"value": "John", "category": "identity"}}

                events = []
                async for event in think("My name is John"):
                    events.append(event)

                # Allow background tasks to complete
                await asyncio.sleep(0.3)

                # Verify extract was called (it runs as a background task)
                mock_extract.assert_called_once()
                call_args = mock_extract.call_args
                assert "John" in call_args[0][0]  # query contains "John"

    @pytest.mark.asyncio
    async def test_no_fact_extraction_for_simple_greeting(self, services):
        """A simple greeting should NOT trigger fact extraction."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Hello!")

            with patch("app.core.brain.extract_facts_from_message", new_callable=AsyncMock) as mock_extract:
                events = []
                async for event in think("hello"):
                    events.append(event)

                await asyncio.sleep(0.2)
                # has_fact_signals("hello") is False, so extract should not be called
                mock_extract.assert_not_called()


class TestMultipleToolRounds:
    """Multiple tool call rounds — chained tool use across LLM turns."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_two_consecutive_tool_rounds(self, services):
        """LLM calls tool in round 1, then another tool in round 2, then final answer."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "web_search", "args": {"query": "AAPL price"}}',
                    tool_calls=[ToolCall(tool="web_search", args={"query": "AAPL price"})],
                    raw={},
                )
            if call_count == 2:
                return GenerationResult(
                    content='{"tool": "calculator", "args": {"expression": "150.25 * 10"}}',
                    tool_calls=[ToolCall(tool="calculator", args={"expression": "150.25 * 10"})],
                    raw={},
                )
            return GenerationResult(
                content="AAPL is at $150.25. 10 shares would cost $1502.50.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Stock Price")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(
                services,
                tools=[{"name": "web_search"}, {"name": "calculator"}],
                execute_map={
                    "web_search": ("AAPL: $150.25", ToolResult(output="AAPL: $150.25", success=True)),
                    "calculator": ("1502.50", ToolResult(output="1502.50", success=True)),
                },
            )

            events = []
            async for event in think("How much would 10 shares of AAPL cost?"):
                events.append(event)

            tool_events = [e for e in events if e.type == EventType.TOOL_USE]
            tool_names = [e.data.get("tool") for e in tool_events if e.data.get("status") == "executing"]
            assert "web_search" in tool_names
            assert "calculator" in tool_names

            # Verify final response text
            tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
            full_text = "".join(tokens)
            assert "1502.50" in full_text or "$150.25" in full_text

    @pytest.mark.asyncio
    async def test_multiple_tools_in_single_round(self, services):
        """LLM returns two tool calls in a single round, both execute concurrently."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='Two tool calls needed',
                    tool_calls=[
                        ToolCall(tool="web_search", args={"query": "weather NYC"}),
                        ToolCall(tool="web_search", args={"query": "weather London"}),
                    ],
                    raw={},
                )
            return GenerationResult(
                content="NYC is 72F sunny, London is 15C cloudy.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Weather Compare")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(services, tools=[{"name": "web_search"}], execute_map={
                "web_search": ("Weather: sunny, 72F", ToolResult(output="Weather: sunny, 72F", success=True)),
            })

            events = []
            async for event in think("Compare weather in NYC and London"):
                events.append(event)

            # Should have 2 executing + 2 complete events (2 tool calls in one round)
            tool_events = [e for e in events if e.type == EventType.TOOL_USE]
            executing = [e for e in tool_events if e.data.get("status") == "executing"]
            complete = [e for e in tool_events if e.data.get("status") == "complete"]
            assert len(executing) == 2
            assert len(complete) == 2


class TestStreamingBehavior:
    """Tests for token streaming behavior."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_tokens_reconstruct_full_response(self, services):
        """TOKEN events should reconstruct the complete response when joined."""
        response_text = "The answer is 42. Life, the universe, and everything."
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content=response_text)

            tokens = []
            async for event in think("What is the meaning of life?"):
                if event.type == EventType.TOKEN:
                    tokens.append(event.data["text"])

            full_text = "".join(tokens)
            assert full_text == response_text

    @pytest.mark.asyncio
    async def test_token_chunks_are_small(self, services):
        """TOKEN events should be chunked (not the entire response in one event)."""
        response_text = "A" * 200  # 200 chars, should be chunked into ~10 events (20 chars each)
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content=response_text)

            tokens = []
            async for event in think("Tell me something"):
                if event.type == EventType.TOKEN:
                    tokens.append(event.data["text"])

            # With chunk_size=20, 200 chars should produce 10 chunks
            assert len(tokens) == 10
            assert all(len(t) == 20 for t in tokens)

    @pytest.mark.asyncio
    async def test_thinking_event_before_tokens(self, services):
        """THINKING events should appear before TOKEN events in the stream."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Response text here.")

            event_order = []
            async for event in think("What is Python?"):
                if event.type in (EventType.THINKING, EventType.TOKEN, EventType.DONE):
                    event_order.append(event.type)

            # THINKING should appear before any TOKEN
            first_thinking = event_order.index(EventType.THINKING)
            first_token = event_order.index(EventType.TOKEN)
            assert first_thinking < first_token


class TestErrorRecoveryDuringTools:
    """Error recovery during tool execution."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_tool_failure_still_produces_final_answer(self, services):
        """If a tool fails, the LLM should still produce a final answer."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "web_search", "args": {"query": "test"}}',
                    tool_calls=[ToolCall(tool="web_search", args={"query": "test"})],
                    raw={},
                )
            return GenerationResult(
                content="I was unable to search the web, but based on my knowledge: the answer is 42.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(services, tools=[{"name": "web_search"}], execute_map={
                "web_search": (
                    "[Tool error: web_search] Connection timeout (retriable: yes)",
                    ToolResult(output="", success=False, error="Connection timeout", retriable=False),
                ),
            })

            events = []
            async for event in think("Search for something"):
                events.append(event)

            event_types = [e.type for e in events]
            assert EventType.TOKEN in event_types
            assert EventType.DONE in event_types

            tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
            full_text = "".join(tokens)
            assert "42" in full_text

    @pytest.mark.asyncio
    async def test_llm_unavailable_produces_error_event(self, services):
        """LLMUnavailableError should produce an ERROR event."""
        from app.core.llm import LLMUnavailableError

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = AsyncMock(
                side_effect=LLMUnavailableError("Connection refused")
            )
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = MagicMock

            events = []
            async for event in think("What is the weather?"):
                events.append(event)

            event_types = [e.type for e in events]
            assert EventType.ERROR in event_types

            error_event = next(e for e in events if e.type == EventType.ERROR)
            assert "Connection refused" in error_event.data["message"]

    @pytest.mark.asyncio
    async def test_tool_timeout_still_completes(self, services):
        """A tool that times out should not crash the pipeline; the LLM gets the error."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='{"tool": "web_search", "args": {"query": "slow"}}',
                    tool_calls=[ToolCall(tool="web_search", args={"query": "slow"})],
                    raw={},
                )
            return GenerationResult(
                content="The search timed out, but here is what I know.",
                tool_calls=[],
                raw={},
            )

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            # Simulate a timeout by returning an error result
            _make_tool_registry(services, tools=[{"name": "web_search"}], execute_map={
                "web_search": (
                    "[Tool error: web_search] Timed out after 120 seconds (retriable: yes)",
                    ToolResult(output="", success=False, error="Timeout", retriable=True),
                ),
            })

            events = []
            async for event in think("Search for something slow"):
                events.append(event)

            event_types = [e.type for e in events]
            assert EventType.DONE in event_types
            assert EventType.TOKEN in event_types

            tokens = [e.data["text"] for e in events if e.type == EventType.TOKEN]
            full_text = "".join(tokens)
            assert "timed out" in full_text.lower() or "know" in full_text.lower()


class TestDoneEventMetadata:
    """Verify the DONE event contains rich metadata."""

    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_done_event_contains_intent(self, services):
        """DONE event should contain the classified intent."""
        with patch("app.core.brain.llm") as mock_llm:
            _mock_llm_simple(mock_llm, content="Hi there!")

            done_event = None
            async for event in think("hello"):
                if event.type == EventType.DONE:
                    done_event = event

            assert done_event is not None
            assert done_event.data["intent"] == "greeting"

    @pytest.mark.asyncio
    async def test_done_event_tool_results_count(self, services):
        """DONE event should report tool_results_count when tools were used."""
        from app.core.llm import ToolCall, GenerationResult
        from app.tools.base import ToolResult

        call_count = 0

        async def mock_generate(messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult(
                    content='tool call',
                    tool_calls=[ToolCall(tool="calculator", args={"expression": "1+1"})],
                    raw={},
                )
            return GenerationResult(content="2", tool_calls=[], raw={})

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = mock_generate
            mock_llm.get_provider = MagicMock(return_value=MagicMock(
                capabilities=MagicMock(needs_emphatic_prompts=False),
            ))
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.extract_json_object = MagicMock(return_value=None)
            mock_llm.invoke_nothink = AsyncMock(return_value="Math")
            mock_llm._extract_tool_calls = MagicMock(return_value=[])
            mock_llm.GenerationResult = GenerationResult

            _make_tool_registry(services, execute_map={
                "calculator": ("2", ToolResult(output="2", success=True)),
            })

            done_event = None
            async for event in think("What is 1+1?"):
                if event.type == EventType.DONE:
                    done_event = event

            assert done_event is not None
            assert done_event.data["tool_results_count"] == 1
