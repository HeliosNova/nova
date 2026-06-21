"""Self-improvement health metrics + dead-cruft pruning (2026-06-13).

The loop emitted skills/auto-tools/lessons with no measurement of whether the
output was useful, and never-used aged artifacts (no failures) escaped the
success-based gate. These pin the health snapshot and the prune.
"""
from app.core import self_improvement as si


def _seed(db):
    # 3 skills: one used, one dead+aged (never used, old), one new+unused.
    db.execute("INSERT INTO skills (name,trigger_pattern,steps,times_used,success_rate,enabled,created_at) "
               "VALUES ('used','p','[]',5,0.8,1,datetime('now','-40 days'))")
    db.execute("INSERT INTO skills (name,trigger_pattern,steps,times_used,success_rate,enabled,created_at) "
               "VALUES ('dead','p','[]',0,0.0,1,datetime('now','-40 days'))")
    db.execute("INSERT INTO skills (name,trigger_pattern,steps,times_used,success_rate,enabled,created_at) "
               "VALUES ('fresh','p','[]',0,0.0,1,datetime('now','-1 days'))")
    # auto-tools: one used, one dead+aged.
    db.execute("INSERT INTO custom_tools (name,description,parameters,code,times_used,success_rate,enabled,created_at) "
               "VALUES ('t_used','d','{}','x',3,0.9,1,datetime('now','-40 days'))")
    db.execute("INSERT INTO custom_tools (name,description,parameters,code,times_used,success_rate,enabled,created_at) "
               "VALUES ('t_dead','d','{}','x',0,0.0,1,datetime('now','-40 days'))")
    # lessons: one helpful, one retrieved-but-never-helpful.
    db.execute("INSERT INTO lessons (topic,correct_answer,confidence,times_retrieved,times_helpful) VALUES ('a','x',0.8,5,3)")
    db.execute("INSERT INTO lessons (topic,correct_answer,confidence,times_retrieved,times_helpful) VALUES ('b','y',0.5,9,0)")


def test_compute_health(db):
    _seed(db)
    h = si.compute_health(db)
    assert h["skills"]["active"] == 3 and h["skills"]["used"] == 1
    assert h["skills"]["dead_unused_aged"] == 1   # 'dead' (old, unused); 'fresh' too new
    assert h["auto_tools"]["used"] == 1 and h["auto_tools"]["dead_unused_aged"] == 1
    assert h["lessons"]["helpful"] == 1
    assert h["lessons"]["retrieved_but_never_helpful"] == 1


def test_prune_disables_dead_aged_only(db):
    _seed(db)
    pruned = si.prune_dead_artifacts(db)
    assert pruned["skills"] == 1 and pruned["auto_tools"] == 1
    # 'used' and 'fresh' survive; 'dead' is disabled.
    enabled = {r["name"] for r in db.fetchall("SELECT name FROM skills WHERE enabled=1")}
    assert "used" in enabled and "fresh" in enabled and "dead" not in enabled
    # Idempotent: second run prunes nothing.
    assert si.prune_dead_artifacts(db) == {"skills": 0, "auto_tools": 0}
