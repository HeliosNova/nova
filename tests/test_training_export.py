"""Tests for the training signal export pipeline (scripts/export_training_signal.py).

Verifies that each signal type extracts data correctly from the DB and
produces valid JSONL output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path for import
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_scripts_dir))

# Skip whole module if the training signal exporter isn't available (e.g. Docker runtime
# image ships only verify_phase_0.py, not the heavyweight training scripts).
_export_available = (_scripts_dir / "export_training_signal.py").is_file()
pytestmark = pytest.mark.skipif(not _export_available, reason="scripts/export_training_signal.py not present")

from app.database import SafeDB


@pytest.fixture
def db(tmp_path):
    """Fresh database with schema + seed data."""
    db_path = str(tmp_path / "test.db")
    _db = SafeDB(db_path)
    # Pre-create kg_facts with valid_from for schema
    conn = _db._get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL, predicate TEXT NOT NULL,
            object TEXT NOT NULL, confidence REAL DEFAULT 0.8,
            source TEXT DEFAULT 'extracted', valid_from TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject, predicate, object)
        )
    """)
    conn.commit()
    _db.init_schema()

    # Seed lessons
    _db.execute(
        "INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("Capital of Australia", "Sydney", "Canberra", "Canberra is the capital, not Sydney", 0.95, "2026-04-15T10:00:00"),
    )
    _db.execute(
        "INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("Python creator", "", "Guido van Rossum", "Created in 1991", 0.8, "2026-04-15T11:00:00"),
    )

    # Seed reflexions
    _db.execute(
        "INSERT INTO reflexions (task_summary, outcome, reflection, quality_score, tools_used, revision_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("What is quantum computing?", "success", "Comprehensive explanation covering qubits, superposition, and entanglement with real-world applications", 0.9, "web_search", 1, "2026-04-15T12:00:00"),
    )
    _db.execute(
        "INSERT INTO reflexions (task_summary, outcome, reflection, quality_score, tools_used, revision_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Calculate the derivative of x^3", "failure", "Failed to use calculator tool, gave wrong mental math answer of 2x^2 instead of 3x^2", 0.2, "calculator", 3, "2026-04-15T13:00:00"),
    )

    # Seed skills (base schema: no 'source' column)
    _db.execute(
        "INSERT INTO skills (name, trigger_pattern, steps, answer_template, success_rate, times_used, enabled, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("Weather Lookup", r"(?i)weather\s+in\s+(\w+)", '[{"tool": "web_search", "args_template": {"query": "weather in {capture_1}"}, "output_key": "result"}]',
         "The weather in {capture_1} is: {result}", 0.85, 5, 1, "2026-04-15T14:00:00"),
    )

    # Seed conversations with tool chains
    _db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("conv-tool-1", "Tool Chain Test"))
    _db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("msg-1", "conv-tool-1", "user", "Find the GDP of Japan", None, "2026-04-15T15:00:00"),
    )
    _db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("msg-2", "conv-tool-1", "tool", "Japan GDP: $4.2 trillion (2024)", "web_search", "2026-04-15T15:00:01"),
    )
    _db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("msg-3", "conv-tool-1", "tool", "Japan GDP growth: 1.2% YoY", "http_fetch", "2026-04-15T15:00:02"),
    )
    _db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("msg-4", "conv-tool-1", "assistant", "Japan's GDP is approximately $4.2 trillion with 1.2% YoY growth.", None, "2026-04-15T15:00:03"),
    )

    yield _db
    _db.close()


class TestCorrectionDPO:
    """Test correction_dpo signal extraction."""

    def test_extracts_from_lessons(self, db):
        from export_training_signal import _export_correction_dpo
        records = _export_correction_dpo(db, None)
        assert len(records) >= 1  # Only lessons with wrong_answer
        dpo_record = records[0]
        assert dpo_record["signal_type"] == "correction_dpo"
        assert dpo_record["prompt"] == "Capital of Australia"
        assert "Canberra" in dpo_record["chosen"]
        assert dpo_record["rejected"] == "Sydney"
        assert dpo_record["metadata"]["confidence"] == 0.95

    def test_extracts_from_jsonl(self, db, tmp_path):
        from export_training_signal import _export_correction_dpo
        jsonl_path = str(tmp_path / "train.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"query": "test q", "chosen": "good", "rejected": "bad", "timestamp": "2026-01-01"}) + "\n")
        records = _export_correction_dpo(db, jsonl_path)
        jsonl_records = [r for r in records if r["metadata"].get("source_table") == "training_data_jsonl"]
        assert len(jsonl_records) == 1
        assert jsonl_records[0]["prompt"] == "test q"

    def test_skips_lessons_without_wrong_answer(self, db):
        from export_training_signal import _export_correction_dpo
        records = _export_correction_dpo(db, None)
        # "Python creator" lesson has no wrong_answer
        topics = [r["prompt"] for r in records]
        assert "Python creator" not in topics


class TestReflexionPositive:
    """Test reflexion_pos signal extraction."""

    def test_extracts_high_quality(self, db):
        from export_training_signal import _export_reflexion_positive
        records = _export_reflexion_positive(db)
        assert len(records) == 1
        assert records[0]["signal_type"] == "reflexion_pos"
        assert records[0]["metadata"]["quality_score"] >= 0.8
        assert "quantum" in records[0]["prompt"].lower()


class TestReflexionNegative:
    """Test reflexion_neg signal extraction."""

    def test_extracts_low_quality(self, db):
        from export_training_signal import _export_reflexion_negative
        records = _export_reflexion_negative(db)
        assert len(records) == 1
        assert records[0]["signal_type"] == "reflexion_neg"
        assert records[0]["metadata"]["quality_score"] < 0.4
        assert "derivative" in records[0]["prompt"].lower()


class TestSkillProcedures:
    """Test skill_procedure signal extraction."""

    def test_extracts_successful_skills(self, db):
        from export_training_signal import _export_skill_procedures
        records = _export_skill_procedures(db)
        assert len(records) == 1
        assert records[0]["signal_type"] == "skill_procedure"
        assert "web_search" in records[0]["chosen"]
        assert records[0]["metadata"]["success_rate"] >= 0.7


class TestReasoningTraces:
    """Test reasoning_trace signal extraction."""

    def test_extracts_critiques(self, db):
        from export_training_signal import _export_reasoning_traces
        records = _export_reasoning_traces(db)
        assert len(records) >= 1
        for r in records:
            assert r["signal_type"] == "reasoning_trace"
            assert "Quality assessment" in r["chosen"]
            assert "Analysis:" in r["chosen"]


class TestToolChains:
    """Test tool_chain signal extraction."""

    def test_extracts_multi_tool_conversations(self, db):
        from export_training_signal import _export_tool_chains
        records = _export_tool_chains(db)
        assert len(records) == 1
        assert records[0]["signal_type"] == "tool_chain"
        assert "GDP" in records[0]["prompt"]
        assert records[0]["metadata"]["tool_count"] == 2


class TestLessonKnowledge:
    """Test lesson_knowledge signal extraction."""

    def test_extracts_high_confidence_lessons(self, db):
        from export_training_signal import _export_lesson_knowledge
        records = _export_lesson_knowledge(db)
        assert len(records) >= 1
        for r in records:
            assert r["signal_type"] == "lesson_knowledge"
            assert r["metadata"]["confidence"] >= 0.6


class TestJSONLFormat:
    """Test that output is valid JSONL."""

    def test_all_records_have_required_fields(self, db):
        from export_training_signal import EXPORTERS, _export_correction_dpo
        all_records = []
        for name, exporter in EXPORTERS.items():
            if name == "correction_dpo":
                all_records.extend(exporter(db, None))
            else:
                all_records.extend(exporter(db))

        for r in all_records:
            assert "signal_type" in r
            assert "prompt" in r
            assert "chosen" in r
            assert "rejected" in r
            assert "metadata" in r
            # Must be JSON-serializable
            json.dumps(r)

    def test_signal_types_valid(self, db):
        from export_training_signal import EXPORTERS
        valid_types = set(EXPORTERS.keys())
        all_records = []
        for name, exporter in EXPORTERS.items():
            if name == "correction_dpo":
                all_records.extend(exporter(db, None))
            else:
                all_records.extend(exporter(db))

        for r in all_records:
            assert r["signal_type"] in valid_types
