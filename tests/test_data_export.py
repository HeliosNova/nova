"""Tests for data_export.py — unified export/import with HMAC signing."""

import json
import os
import tempfile

import pytest

from app.core.data_export import (
    SignatureError,
    canonical_json,
    export_bundle,
    export_kg_fact,
    export_lesson,
    generate_key,
    import_bundle,
    import_kg_fact,
    import_lesson,
    load_key,
    sign_data,
    verify_data,
)


# ---------------------------------------------------------------------------
# Signing primitives
# ---------------------------------------------------------------------------


def test_canonical_json_sorted():
    data = {"z": 1, "a": 2, "m": 3}
    result = canonical_json(data)
    assert result == b'{"a":2,"m":3,"z":1}'


def test_sign_verify_roundtrip():
    key = bytes.fromhex(generate_key())
    data = {"topic": "test", "answer": "42"}
    sig = sign_data(data, key)
    assert verify_data(data, sig, key)


def test_sign_excludes_signature_field():
    key = bytes.fromhex(generate_key())
    data = {"topic": "test", "answer": "42"}
    sig = sign_data(data, key)
    # Adding signature field should not change verification
    data_with_sig = {**data, "signature": sig}
    assert verify_data(data_with_sig, sig, key)


def test_tampered_data_fails():
    key = bytes.fromhex(generate_key())
    data = {"topic": "test", "answer": "42"}
    sig = sign_data(data, key)
    data["answer"] = "43"
    assert not verify_data(data, sig, key)


def test_wrong_key_fails():
    key1 = bytes.fromhex(generate_key())
    key2 = bytes.fromhex(generate_key())
    data = {"topic": "test"}
    sig = sign_data(data, key1)
    assert not verify_data(data, sig, key2)


def test_load_key_from_file():
    key_hex = generate_key()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hex", delete=False) as f:
        f.write(key_hex)
        f.flush()
        loaded = load_key(f.name)
    os.unlink(f.name)
    assert loaded == bytes.fromhex(key_hex)


# ---------------------------------------------------------------------------
# Lesson export/import
# ---------------------------------------------------------------------------


class FakeLesson:
    def __init__(self):
        self.id = 1
        self.topic = "Python"
        self.wrong_answer = "print is a statement"
        self.correct_answer = "print is a function"
        self.lesson_text = "In Python 3, print() is a function, not a statement."
        self.confidence = 0.9
        self.context = "user asked about print"
        self.created_at = "2026-01-01"


def test_export_lesson_from_dataclass():
    lesson = FakeLesson()
    result = export_lesson(lesson)
    assert result["topic"] == "Python"
    assert result["correct_answer"] == "print is a function"
    assert result["version"] == "1.0"
    assert result["author"] == "nova"
    assert "export_timestamp" in result
    assert "signature" not in result


def test_export_lesson_signed():
    key_hex = generate_key()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hex", delete=False) as f:
        f.write(key_hex)
        key_path = f.name

    try:
        lesson = FakeLesson()
        result = export_lesson(lesson, key_path=key_path)
        assert "signature" in result

        # Verify signature
        key = bytes.fromhex(key_hex)
        assert verify_data(result, result["signature"], key)
    finally:
        os.unlink(key_path)


def test_export_lesson_from_dict():
    data = {"topic": "test", "correct_answer": "yes", "lesson_text": "always yes"}
    result = export_lesson(data)
    assert result["topic"] == "test"


# ---------------------------------------------------------------------------
# KG fact export/import
# ---------------------------------------------------------------------------


class FakeFact:
    def __init__(self):
        self.id = 7
        self.subject = "Python"
        self.predicate = "is_a"
        self.object = "programming language"
        self.confidence = 0.95
        self.source = "extracted"
        self.valid_from = "2026-01-01"
        self.valid_to = None
        self.provenance = "conversation_123"
        self.created_at = "2026-01-01"
        self.superseded_by = None


def test_export_kg_fact():
    fact = FakeFact()
    result = export_kg_fact(fact)
    assert result["subject"] == "Python"
    assert result["predicate"] == "is_a"
    assert result["object"] == "programming language"
    assert result["version"] == "1.0"


def test_export_kg_fact_signed():
    key_hex = generate_key()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hex", delete=False) as f:
        f.write(key_hex)
        key_path = f.name

    try:
        fact = FakeFact()
        result = export_kg_fact(fact, key_path=key_path)
        assert "signature" in result
        key = bytes.fromhex(key_hex)
        assert verify_data(result, result["signature"], key)
    finally:
        os.unlink(key_path)
