"""Phase 1 foundation tests — database, config, schema, JSON extraction."""

from __future__ import annotations

import json

import pytest


class TestConfig:
    def test_config_loads_defaults(self):
        from app.config import Config
        c = Config()
        assert c.LLM_MODEL == "qwen3.5:27b"
        assert c.PORT == 8000
        assert c.MAX_TOOL_ROUNDS == 5

    def test_config_reads_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "test-model:7b")
        monkeypatch.setenv("PORT", "9999")
        from app.config import Config
        c = Config()
        assert c.LLM_MODEL == "test-model:7b"
        assert c.PORT == 9999


class TestDatabase:
    def test_schema_creates_tables(self, db):
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {row["name"] for row in tables}
        assert "conversations" in names
        assert "messages" in names
        assert "user_facts" in names
        assert "lessons" in names
        assert "skills" in names
        assert "documents" in names

    def test_insert_and_fetch(self, db):
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-1", "Test Chat"),
        )
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("conv-1",))
        assert row is not None
        assert row["title"] == "Test Chat"

    def test_user_facts_unique_key(self, db):
        db.execute(
            "INSERT INTO user_facts (key, value) VALUES (?, ?)",
            ("name", "Alice"),
        )
        with pytest.raises(Exception):
            db.execute(
                "INSERT INTO user_facts (key, value) VALUES (?, ?)",
                ("name", "Bob"),
            )

    def test_fts5_table_exists(self, db):
        # FTS5 virtual tables appear in sqlite_master
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert len(tables) == 1

    def test_fts5_insert_and_search(self, db):
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            ("c1", "d1", "Bitcoin halving occurs approximately every four years"),
        )
        results = db.fetchall(
            "SELECT * FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("bitcoin halving",),
        )
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"


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

    def test_extract_tool_call_alternative_keys(self):
        from app.core.llm import _extract_tool_call
        tools = [{"name": "calculator"}]
        content = '{"name": "calculator", "arguments": {"expression": "2+2"}}'
        tc = _extract_tool_call(content, tools)
        assert tc is not None
        assert tc.tool == "calculator"
        assert tc.args == {"expression": "2+2"}


class TestStripThinkTags:
    def test_strips_think_blocks(self):
        from app.core.llm import _strip_think_tags
        text = "<think>internal reasoning here</think>\nThe answer is 42."
        assert _strip_think_tags(text).strip() == "The answer is 42."

    def test_strips_multiline_think(self):
        from app.core.llm import _strip_think_tags
        text = "<think>\nstep 1\nstep 2\nstep 3\n</think>\nFinal answer."
        assert _strip_think_tags(text).strip() == "Final answer."

    def test_no_think_tags_unchanged(self):
        from app.core.llm import _strip_think_tags
        text = "Just a normal response."
        assert _strip_think_tags(text) == "Just a normal response."


class TestSchema:
    def test_stream_event_to_sse(self):
        from app.schema import StreamEvent, EventType
        event = StreamEvent(type=EventType.TOKEN, data={"text": "hello"})
        sse = event.to_sse()
        assert "event: token" in sse
        assert '"text": "hello"' in sse

    def test_chat_request_defaults(self):
        from app.schema import ChatRequest
        req = ChatRequest(query="hello")
        assert req.conversation_id is None
        assert req.image_base64 is None

    def test_health_response(self):
        from app.schema import HealthResponse
        h = HealthResponse(ollama_connected=True, db_connected=True)
        assert h.status == "ok"


# ===========================================================================
# Config Validation (Item 11)
# ===========================================================================

class TestConfigValidation:
    def test_valid_config_no_warnings(self):
        from app.config import Config
        c = Config()
        warnings = c.validate()
        assert warnings == []

    def test_invalid_ollama_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "ftp://bad.url")
        from app.config import Config
        c = Config()
        warnings = c.validate()
        assert any("OLLAMA_URL" in w for w in warnings)

    def test_invalid_port(self, monkeypatch):
        monkeypatch.setenv("PORT", "99999")
        from app.config import Config
        c = Config()
        warnings = c.validate()
        assert any("PORT" in w for w in warnings)

    def test_invalid_searxng_url(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "not-a-url")
        from app.config import Config
        c = Config()
        warnings = c.validate()
        assert any("SEARXNG_URL" in w for w in warnings)
