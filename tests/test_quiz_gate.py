"""Lesson Quiz technique-lesson gate (2026-08-27).

29 of 44 live lessons store a process instruction ("do a web search", "use
the calculator") as their correct_answer. The quiz generator has no fact to
ask for, so it invented new problems (fresh arithmetic, provenance-metadata
questions) that the grader then failed against the unrelated stored answer —
structural false failures feeding lesson-confidence drops and curiosity
re-research loops (the art-history churn family).
"""
from app.monitors.heartbeat_loop import _UNQUIZZABLE_ANSWER_RE


INSTRUCTION_ANSWERS = [
    "To answer factual art history questions, perform a web search for the specific artwork.",
    "Use the calculator tool for arithmetic instead of computing mentally.",
    "Use a web search to verify current facts before answering.",
    "You should look it up rather than relying on memory.",
    "Consult current sources when asked about recent events.",
    "Verify with the calculator before stating numeric results.",
]

FACTUAL_ANSWERS = [
    "Nova's internal task scheduler is codenamed Chronos.",
    "The Mona Lisa was painted by Leonardo da Vinci around 1503-1506.",
    "Water boils at 100 degrees Celsius at sea-level pressure.",
    "The user's preferred language for scripting is Python.",
    "SQLite 3.53.4 is baked into the image because Debian's 3.46.1 carries a WAL-reset bug.",
]


class TestUnquizzableGate:
    def test_instruction_answers_gated(self):
        for a in INSTRUCTION_ANSWERS:
            assert _UNQUIZZABLE_ANSWER_RE.search(a), a

    def test_factual_answers_pass(self):
        for a in FACTUAL_ANSWERS:
            assert not _UNQUIZZABLE_ANSWER_RE.search(a), a
