"""Goal derivation excludes junk keywords (audit 2026-08-23).

The capability-gap clusterer minted goals from stopword-ish keywords ('does',
'higher', 'clock') — which produced 3 dead goals that could never succeed. It now
filters STOP_WORDS + _GOAL_KEYWORD_JUNK before clustering; substantive topics
still cluster into goals.
"""

from app.core.goal_deriver import _derive_goals_sync


def _seed_gap(db, query):
    db.execute(
        "INSERT INTO capability_gaps (query, reason, reviewed) VALUES (?, 'test', 0)",
        (query,),
    )


def test_junk_keywords_do_not_become_goals(db):
    # 4 failures all sharing only the junk words 'higher'/'does' — must NOT cluster.
    for i in range(4):
        _seed_gap(db, f"does the higher option work here {i}")
    goals = _derive_goals_sync(db, max_new_goals=5)
    joined = " ".join(g.get("goal", "").lower() for g in goals)
    assert "higher" not in joined and "does" not in joined, f"junk goal minted: {joined!r}"


def test_substantive_keyword_still_clusters(db):
    # A real topic repeated 4x SHOULD be eligible to mint a capability-gap goal.
    for i in range(4):
        _seed_gap(db, f"semiconductor supply forecast {i}")
    goals = _derive_goals_sync(db, max_new_goals=5)
    joined = " ".join(g.get("goal", "").lower() for g in goals)
    assert "semiconductor" in joined, f"substantive keyword was filtered out: {joined!r}"
