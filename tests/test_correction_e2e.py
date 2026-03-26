"""Item 61: Test correction handling with history walking.

Mock a conversation where the assistant gave a wrong answer, user corrects it,
verify the correct assistant message is identified and a lesson is created.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services, think, _run_post_processing
from app.core.learning import LearningEngine, is_likely_correction, response_pushes_back, Correction
from app.core.memory import ConversationStore, UserFactStore
from app.schema import EventType


@pytest.fixture
def services(db):
    """Set up services with learning engine for correction tests."""
    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
        learning=LearningEngine(db),
    )
    set_services(svc)
    return svc


class TestIsLikelyCorrection:
    """Test the regex pre-filter for correction detection."""

    def test_actually_detected(self):
        assert is_likely_correction("Actually, Python was created by Guido")

    def test_thats_wrong_detected(self):
        assert is_likely_correction("That's wrong, it should be X")

    def test_youre_wrong_detected(self):
        assert is_likely_correction("You're wrong about that")

    def test_correct_answer_is_detected(self):
        assert is_likely_correction("The correct answer is 42")

    def test_it_should_be_detected(self):
        assert is_likely_correction("It should be lowercase")

    def test_normal_question_not_detected(self):
        assert not is_likely_correction("What is the capital of France?")

    def test_greeting_not_detected(self):
        assert not is_likely_correction("Hello, how are you?")


class TestCorrectionHistoryWalking:
    """Test that correction detection properly walks conversation history."""

    @pytest.mark.asyncio
    async def test_correction_detects_previous_wrong_answer(self, services):
        """Walk history, find wrong assistant message, detect correction."""
        # Build a conversation: user asks, assistant gives wrong answer
        conv_id = services.conversations.create_conversation("test")
        services.conversations.add_message(conv_id, "user", "Who created Python?")
        services.conversations.add_message(conv_id, "assistant", "Python was created by James Gosling.")

        # Now the correction cycle: brain.think() saves its response first (step 11),
        # then the correction handler runs (step 13) and skips 1 assistant message.
        # We simulate this by adding the new assistant message before calling _run_post_processing.
        new_response = "I apologize. Python was created by Guido van Rossum."
        services.conversations.add_message(conv_id, "user", "Actually, Python was created by Guido van Rossum")
        services.conversations.add_message(conv_id, "assistant", new_response)

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"is_correction": true, "topic": "Python creator", "wrong_answer": "Python was created by James Gosling", "correct_answer": "Python was created by Guido van Rossum", "lesson_text": "Python was created by Guido van Rossum, not James Gosling"}')
            mock_llm.extract_json_object = lambda x: {
                "is_correction": True,
                "topic": "Python creator",
                "wrong_answer": "Python was created by James Gosling",
                "correct_answer": "Python was created by Guido van Rossum",
                "lesson_text": "Python was created by Guido van Rossum, not James Gosling",
            }

            events = []
            async for event in _run_post_processing(
                svc=services,
                query="Actually, Python was created by Guido van Rossum",
                final_content=new_response,
                intent="correction",
                conversation_id=conv_id,
                tool_results=[],
                matched_skill=None,
                used_lesson_ids=[],
                is_error=False,
                reflexion_quality=None,
                reflexion_reason="",
            ):
                events.append(event)

            # Should have a lesson_learned event
            lesson_events = [e for e in events if e.type == EventType.LESSON_LEARNED]
            assert len(lesson_events) == 1
            assert lesson_events[0].data["topic"] == "Python creator"

    @pytest.mark.asyncio
    async def test_correction_creates_lesson_in_db(self, services):
        """Correction should persist a lesson to the database."""
        conv_id = services.conversations.create_conversation("test")
        services.conversations.add_message(conv_id, "user", "What is the capital of Australia?")
        services.conversations.add_message(conv_id, "assistant", "The capital of Australia is Sydney.")
        services.conversations.add_message(conv_id, "user", "That's wrong, the capital is Canberra")
        services.conversations.add_message(conv_id, "assistant", "You're right, the capital of Australia is Canberra.")

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"is_correction": true, "topic": "Capital of Australia", "wrong_answer": "The capital of Australia is Sydney", "correct_answer": "The capital of Australia is Canberra", "lesson_text": "The capital of Australia is Canberra, not Sydney"}')
            mock_llm.extract_json_object = lambda x: {
                "is_correction": True,
                "topic": "Capital of Australia",
                "wrong_answer": "The capital of Australia is Sydney",
                "correct_answer": "The capital of Australia is Canberra",
                "lesson_text": "The capital of Australia is Canberra, not Sydney",
            }

            events = []
            async for event in _run_post_processing(
                svc=services,
                query="That's wrong, the capital is Canberra",
                final_content="You're right, the capital of Australia is Canberra.",
                intent="correction",
                conversation_id=conv_id,
                tool_results=[],
                matched_skill=None,
                used_lesson_ids=[],
                is_error=False,
                reflexion_quality=None,
                reflexion_reason="",
            ):
                events.append(event)

        # Check lesson was saved
        all_lessons = services.learning.get_all_lessons()
        assert len(all_lessons) >= 1
        lesson = all_lessons[0]
        assert "Canberra" in lesson.correct_answer or "Canberra" in (lesson.lesson_text or "")

    @pytest.mark.asyncio
    async def test_no_correction_when_not_detected(self, services):
        """Normal messages should not produce lesson_learned events."""
        conv_id = services.conversations.create_conversation("test")
        services.conversations.add_message(conv_id, "user", "Tell me about Python")
        services.conversations.add_message(conv_id, "assistant", "Python is a programming language.")
        services.conversations.add_message(conv_id, "user", "Thanks!")
        services.conversations.add_message(conv_id, "assistant", "You're welcome!")

        events = []
        async for event in _run_post_processing(
            svc=services,
            query="Thanks!",
            final_content="You're welcome!",
            intent="general",
            conversation_id=conv_id,
            tool_results=[],
            matched_skill=None,
            used_lesson_ids=[],
            is_error=False,
            reflexion_quality=None,
            reflexion_reason="",
        ):
            events.append(event)

        # Should NOT have any lesson_learned events
        lesson_events = [e for e in events if e.type == EventType.LESSON_LEARNED]
        assert len(lesson_events) == 0

    @pytest.mark.asyncio
    async def test_correction_skips_current_assistant_response(self, services):
        """History walking should skip the current assistant response (skip=1)."""
        conv_id = services.conversations.create_conversation("test")
        # Original question + wrong answer
        services.conversations.add_message(conv_id, "user", "What is 2+2?")
        services.conversations.add_message(conv_id, "assistant", "2+2 = 5")
        # Correction + new response (step 11 saves before step 13 runs)
        services.conversations.add_message(conv_id, "user", "No, 2+2 = 4")
        services.conversations.add_message(conv_id, "assistant", "You're right, 2+2 = 4.")

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"is_correction": true, "topic": "2+2", "wrong_answer": "2+2 = 5", "correct_answer": "2+2 = 4", "lesson_text": "2+2 = 4, not 5"}')
            mock_llm.extract_json_object = lambda x: {
                "is_correction": True,
                "topic": "2+2",
                "wrong_answer": "2+2 = 5",
                "correct_answer": "2+2 = 4",
                "lesson_text": "2+2 = 4, not 5",
            }

            # The correction handler's detect_correction receives prev_answer.
            # We verify this indirectly by checking that the correction was processed.
            events = []
            async for event in _run_post_processing(
                svc=services,
                query="No, 2+2 = 4",
                final_content="You're right, 2+2 = 4.",
                intent="correction",
                conversation_id=conv_id,
                tool_results=[],
                matched_skill=None,
                used_lesson_ids=[],
                is_error=False,
                reflexion_quality=None,
                reflexion_reason="",
            ):
                events.append(event)

            lesson_events = [e for e in events if e.type == EventType.LESSON_LEARNED]
            assert len(lesson_events) == 1

            # Verify the mock was called with the wrong answer (2+2 = 5),
            # not the corrected answer
            call_args = mock_llm.invoke_nothink.call_args
            prompt_content = call_args[0][0][1]["content"]  # user message
            assert "2+2 = 5" in prompt_content


class TestResponsePushesBack:
    """Test the pushback detection for Nova's responses."""

    def test_actually_the_capital_is(self):
        assert response_pushes_back(
            "Actually, the capital of Australia is Canberra, not Sydney."
        )

    def test_stand_by_my_answer(self):
        assert response_pushes_back(
            "I need to stand by my original answer here."
        )

    def test_my_original_answer(self):
        assert response_pushes_back(
            "My original answer was correct. The capital is Canberra."
        )

    def test_answer_is_correct(self):
        assert response_pushes_back(
            "The answer is correct — Canberra is indeed the capital."
        )

    def test_i_respectfully_disagree(self):
        assert response_pushes_back(
            "I respectfully disagree. According to multiple sources, Canberra is the capital."
        )

    def test_search_confirms(self):
        assert response_pushes_back(
            "My web search results confirm that Canberra is the capital of Australia."
        )

    def test_common_misconception(self):
        assert response_pushes_back(
            "This is a common misconception. Sydney is the largest city, but Canberra is the capital."
        )

    def test_i_can_confirm(self):
        assert response_pushes_back(
            "I can confirm that the capital of Australia is Canberra."
        )

    def test_however_the_capital_is(self):
        assert response_pushes_back(
            "However, the capital of Australia is Canberra, not Sydney."
        )

    def test_normal_agreement_not_pushback(self):
        assert not response_pushes_back(
            "You're right, the capital of Australia is Canberra."
        )

    def test_apology_not_pushback(self):
        assert not response_pushes_back(
            "I apologize for the error. Python was created by Guido van Rossum."
        )

    def test_thanks_not_pushback(self):
        assert not response_pushes_back(
            "Thank you for the correction! I'll remember that."
        )

    def test_empty_not_pushback(self):
        assert not response_pushes_back("")

    def test_none_not_pushback(self):
        assert not response_pushes_back(None)


class TestCorrectionPushbackGuard:
    """Test that corrections are NOT saved when Nova pushes back."""

    @pytest.mark.asyncio
    async def test_no_lesson_when_nova_pushes_back(self, services):
        """If Nova's response disagrees with the user's correction, skip saving."""
        conv_id = services.conversations.create_conversation("test")
        services.conversations.add_message(conv_id, "user", "What is the capital of Australia?")
        services.conversations.add_message(conv_id, "assistant", "The capital of Australia is Canberra.")
        # User incorrectly "corrects" Nova
        services.conversations.add_message(
            conv_id, "user", "No that's wrong, the capital of Australia is Sydney"
        )
        # Nova pushes back — correctly stands its ground
        pushback_response = (
            "Actually, the capital of Australia is Canberra, not Sydney. "
            "Sydney is the largest city, but Canberra has been the capital since 1913."
        )
        services.conversations.add_message(conv_id, "assistant", pushback_response)

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=(
                '{"is_correction": true, "topic": "Capital of Australia", '
                '"wrong_answer": "The capital of Australia is Canberra", '
                '"correct_answer": "The capital of Australia is Sydney", '
                '"lesson_text": "The capital of Australia is Sydney, not Canberra"}'
            ))
            mock_llm.extract_json_object = lambda x: {
                "is_correction": True,
                "topic": "Capital of Australia",
                "wrong_answer": "The capital of Australia is Canberra",
                "correct_answer": "The capital of Australia is Sydney",
                "lesson_text": "The capital of Australia is Sydney, not Canberra",
            }

            events = []
            async for event in _run_post_processing(
                svc=services,
                query="No that's wrong, the capital of Australia is Sydney",
                final_content=pushback_response,
                intent="correction",
                conversation_id=conv_id,
                tool_results=[],
                matched_skill=None,
                used_lesson_ids=[],
                is_error=False,
                reflexion_quality=None,
                reflexion_reason="",
            ):
                events.append(event)

            # Should NOT have any lesson_learned events — Nova was right
            lesson_events = [e for e in events if e.type == EventType.LESSON_LEARNED]
            assert len(lesson_events) == 0

            # Verify no lesson was saved to the database
            all_lessons = services.learning.get_all_lessons()
            assert len(all_lessons) == 0

    @pytest.mark.asyncio
    async def test_lesson_saved_when_nova_agrees(self, services):
        """If Nova agrees with the correction, lesson should be saved normally."""
        conv_id = services.conversations.create_conversation("test")
        services.conversations.add_message(conv_id, "user", "Who created Python?")
        services.conversations.add_message(conv_id, "assistant", "Python was created by James Gosling.")
        services.conversations.add_message(
            conv_id, "user", "Actually, Python was created by Guido van Rossum"
        )
        # Nova agrees — the correction was valid
        agreement_response = "You're right, I apologize. Python was created by Guido van Rossum."
        services.conversations.add_message(conv_id, "assistant", agreement_response)

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=(
                '{"is_correction": true, "topic": "Python creator", '
                '"wrong_answer": "Python was created by James Gosling", '
                '"correct_answer": "Python was created by Guido van Rossum", '
                '"lesson_text": "Python was created by Guido van Rossum, not James Gosling"}'
            ))
            mock_llm.extract_json_object = lambda x: {
                "is_correction": True,
                "topic": "Python creator",
                "wrong_answer": "Python was created by James Gosling",
                "correct_answer": "Python was created by Guido van Rossum",
                "lesson_text": "Python was created by Guido van Rossum, not James Gosling",
            }

            events = []
            async for event in _run_post_processing(
                svc=services,
                query="Actually, Python was created by Guido van Rossum",
                final_content=agreement_response,
                intent="correction",
                conversation_id=conv_id,
                tool_results=[],
                matched_skill=None,
                used_lesson_ids=[],
                is_error=False,
                reflexion_quality=None,
                reflexion_reason="",
            ):
                events.append(event)

            # Should have a lesson_learned event — Nova agreed the correction was valid
            lesson_events = [e for e in events if e.type == EventType.LESSON_LEARNED]
            assert len(lesson_events) == 1
            assert lesson_events[0].data["topic"] == "Python creator"

            # Verify lesson was saved to the database
            all_lessons = services.learning.get_all_lessons()
            assert len(all_lessons) >= 1
