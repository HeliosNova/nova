"""Tests for app.core.grpo_dataset — RLVR signals → GRPO groups + DPO pairs."""

from __future__ import annotations

import json
import pytest

from app.core import grpo_dataset, rlvr
from app.database import get_db, _instances


@pytest.fixture
def signals_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "grpo.db"))
    _instances.clear()
    db = get_db()
    db.init_schema()
    yield db
    db.close()
    _instances.clear()


def _seed(query: str, response: str, value: float, signal_type="tool_correct"):
    rlvr.record_signal(
        signal_type, value, query=query, response=response,
        evidence="test", conversation_id="conv-1",
    )


# ---- _normalize_query ---------------------------------------------------

def test_normalize_query_collapses_whitespace():
    assert grpo_dataset._normalize_query("  Hello   World  ") == "hello world"


def test_normalize_query_strips_trailing_punctuation():
    assert grpo_dataset._normalize_query("How are you?") == "how are you"


def test_normalize_query_empty():
    assert grpo_dataset._normalize_query("") == ""
    assert grpo_dataset._normalize_query("   ") == ""


# ---- build_groups -------------------------------------------------------

def test_build_groups_groups_by_query(signals_db):
    _seed("calculate 2 + 2", "4", 1.0)
    _seed("calculate 2 + 2", "5", 0.0)
    _seed("what is python", "a programming language", 1.0)

    groups = grpo_dataset.build_groups(min_group_size=1)
    by_prompt = {g.prompt: g for g in groups}
    # Two unique queries
    assert "calculate 2 + 2" in by_prompt
    assert "what is python" in by_prompt
    # First group has 2 completions
    assert len(by_prompt["calculate 2 + 2"].completions) == 2
    assert len(by_prompt["what is python"].completions) == 1


def test_build_groups_separates_by_signal_type(signals_db):
    _seed("foo", "bar", 1.0, signal_type="tool_correct")
    _seed("foo", "baz", 0.5, signal_type="claim_grounded")
    groups = grpo_dataset.build_groups(min_group_size=1)
    types = sorted(g.signal_type for g in groups)
    assert types == ["claim_grounded", "tool_correct"]


def test_build_groups_min_size_filter(signals_db):
    _seed("only one", "a", 1.0)
    _seed("two", "a", 1.0)
    _seed("two", "b", 0.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    prompts = sorted(g.prompt for g in groups)
    assert prompts == ["two"]


def test_build_groups_empty_db(signals_db):
    assert grpo_dataset.build_groups() == []


def test_build_groups_only_unconsumed(signals_db):
    _seed("x", "a", 1.0)
    _seed("x", "b", 0.0)
    rows = rlvr.query_signals("tool_correct")
    rlvr.mark_consumed([r.id for r in rows])
    assert grpo_dataset.build_groups(only_unconsumed=True) == []
    g = grpo_dataset.build_groups(only_unconsumed=False)
    assert len(g) == 1


def test_build_groups_drops_empty_responses(signals_db):
    _seed("q", "real answer", 1.0)
    _seed("q", "", 0.0)  # empty response — dropped
    _seed("q", "   ", 0.5)  # whitespace-only — dropped
    groups = grpo_dataset.build_groups(min_group_size=1)
    assert len(groups) == 1
    assert len(groups[0].completions) == 1


# ---- compute_advantages -------------------------------------------------

def test_advantages_zero_for_uniform_group():
    g = grpo_dataset.GRPOGroup(
        prompt="x", signal_type="tool_correct",
        completions=["a", "b", "c"], rewards=[0.5, 0.5, 0.5],
    )
    g.compute_advantages()
    assert all(abs(a) < 1e-6 for a in g.advantages)


def test_advantages_standardize_to_zero_mean():
    g = grpo_dataset.GRPOGroup(
        prompt="x", signal_type="tool_correct",
        completions=["a", "b"], rewards=[0.0, 1.0],
    )
    g.compute_advantages()
    assert abs(sum(g.advantages)) < 1e-6


def test_is_trainable_zero_variance_excluded():
    g = grpo_dataset.GRPOGroup(
        prompt="x", signal_type="tool_correct",
        completions=["a", "b"], rewards=[0.5, 0.5],
    )
    assert g.is_trainable() is False


def test_is_trainable_size_threshold():
    g = grpo_dataset.GRPOGroup(
        prompt="x", signal_type="tool_correct",
        completions=["a"], rewards=[1.0],
    )
    assert g.is_trainable(min_size=2) is False


# ---- to_grpo_dataset ----------------------------------------------------

def test_to_grpo_dataset_flat_shape(signals_db):
    _seed("calc", "4", 1.0)
    _seed("calc", "5", 0.0)
    _seed("calc", "4.0", 1.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    flat = grpo_dataset.to_grpo_dataset(groups)
    assert len(flat["prompt"]) == 3
    assert len(flat["completion"]) == 3
    assert len(flat["reward"]) == 3
    assert len(flat["advantage"]) == 3
    # advantages sum to ~0
    assert abs(sum(flat["advantage"])) < 1e-6


def test_to_grpo_dataset_filters_untrainable(signals_db):
    _seed("uniform", "a", 0.5)
    _seed("uniform", "b", 0.5)  # zero variance — untrainable
    _seed("varied", "a", 0.0)
    _seed("varied", "b", 1.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    flat = grpo_dataset.to_grpo_dataset(groups, require_trainable=True)
    # only the "varied" group should appear
    assert all(p == "varied" for p in flat["prompt"])
    assert len(flat["prompt"]) == 2


# ---- to_dpo_pairs -------------------------------------------------------

def test_to_dpo_pairs_picks_best_and_worst(signals_db):
    _seed("Q", "good answer", 1.0)
    _seed("Q", "ok answer", 0.5)
    _seed("Q", "bad answer", 0.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    pairs = grpo_dataset.to_dpo_pairs(groups)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["chosen"] == "good answer"
    assert p["rejected"] == "bad answer"
    assert p["chosen_reward"] == 1.0
    assert p["rejected_reward"] == 0.0


def test_to_dpo_pairs_skips_small_gap(signals_db):
    _seed("Q", "ok1", 0.6)
    _seed("Q", "ok2", 0.5)
    groups = grpo_dataset.build_groups(min_group_size=2)
    pairs = grpo_dataset.to_dpo_pairs(groups, min_reward_gap=0.5)
    assert pairs == []


def test_to_dpo_pairs_skips_identical_completions(signals_db):
    _seed("Q", "same", 1.0)
    _seed("Q", "same", 0.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    pairs = grpo_dataset.to_dpo_pairs(groups)
    assert pairs == []


# ---- write_jsonl --------------------------------------------------------

def test_write_jsonl(tmp_path):
    items = [{"a": 1}, {"a": 2, "b": [1, 2]}]
    path = tmp_path / "out.jsonl"
    n = grpo_dataset.write_jsonl(items, str(path))
    assert n == 2
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}


# ---- stats --------------------------------------------------------------

def test_stats_reports_counts(signals_db):
    for i in range(5):
        _seed("q1", f"r{i}", 1.0 if i % 2 else 0.0)
    _seed("q2", "x", 1.0)
    _seed("q2", "y", 0.0)
    groups = grpo_dataset.build_groups(min_group_size=2)
    s = grpo_dataset.stats(groups)
    assert s["n_groups"] == 2
    assert s["n_trainable"] == 2
    assert s["n_completions"] == 7
    assert "tool_correct" in s["by_signal_type"]
