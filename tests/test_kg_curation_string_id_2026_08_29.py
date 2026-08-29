"""KG LLM curation aborted its whole batch on a string "id".

Live 2026-08-29: `KG LLM curation failed (heuristic pass still ran):
'<=' not supported between instances of 'int' and 'str'` — the 9B returned
{"id": "1"} instead of {"id": 1}, so `1 <= idx` raised TypeError inside the
result loop and the surrounding `except` discarded the ENTIRE batch. Effect:
0 successes / 1 failure in 48h, i.e. LLM garbage-retirement never actually
ran and the heuristic pass masked the outage.

Two-layer fix, both asserted here: a json_schema pinning id to integer and
verdict to an enum, plus an int() coercion so one malformed element is skipped
rather than aborting its siblings.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.kg import KnowledgeGraph


@pytest.mark.asyncio
async def test_string_id_does_not_abort_the_batch(db):
    """A string "id" must still retire that fact, not kill the whole pass."""
    kg = KnowledgeGraph(db)
    await kg.add_fact("testsubject", "is", "obvious garbage", confidence=0.1)

    from app.core import llm as llm_mod

    # Exactly what the 9B emitted live: id as a STRING.
    payload = '{"results": [{"id": "1", "verdict": "garbage"}]}'
    with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as inv, \
         patch.object(llm_mod, "extract_json_object") as ext:
        inv.return_value = payload
        ext.side_effect = lambda raw: json.loads(raw)
        res = await kg.curate(sample_size=20, heuristic=False)

    # Before the fix this raised TypeError, was swallowed, and returned llm=0.
    assert res["llm"] >= 1, "string id still aborts the curation batch"


@pytest.mark.asyncio
async def test_curation_request_pins_a_schema(db):
    """The real fix is structural: constrain the output shape."""
    kg = KnowledgeGraph(db)
    await kg.add_fact("testsubject", "is", "obvious garbage", confidence=0.1)

    from app.core import llm as llm_mod

    with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as inv, \
         patch.object(llm_mod, "extract_json_object") as ext:
        inv.return_value = '{"results": []}'
        ext.side_effect = lambda raw: json.loads(raw)
        await kg.curate(sample_size=20, heuristic=False)

        assert inv.await_count >= 1
        schema = inv.await_args.kwargs.get("json_schema")
        assert schema, "curation must pin a json_schema"
        item = schema["properties"]["results"]["items"]["properties"]
        assert item["id"]["type"] == "integer", "id must be pinned to integer"
        assert set(item["verdict"]["enum"]) == {"keep", "garbage"}


@pytest.mark.asyncio
async def test_one_bad_element_does_not_drop_its_siblings(db):
    """A junk element is skipped; the valid one beside it still applies."""
    kg = KnowledgeGraph(db)
    await kg.add_fact("testsubject", "is", "obvious garbage", confidence=0.1)

    from app.core import llm as llm_mod

    payload = ('{"results": [{"id": "not-a-number", "verdict": "garbage"}, '
               '{"id": 1, "verdict": "garbage"}]}')
    with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as inv, \
         patch.object(llm_mod, "extract_json_object") as ext:
        inv.return_value = payload
        ext.side_effect = lambda raw: json.loads(raw)
        res = await kg.curate(sample_size=20, heuristic=False)

    assert res["llm"] >= 1, "an unparseable id aborted the surviving sibling"
