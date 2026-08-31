"""Silent errors made loud — regression locks from the 2026-08-31 sweep.

An AST sweep found 230 trivial exception swallows, 80 debug-hidden handlers
and 41 silent JSON eats. Most are deliberate best-effort; these two changed
BEHAVIOUR-relevant outcomes silently and are now loud:

1. config._load_overrides: a corrupt config_overrides.json was swallowed
   entirely — Nova boots on DEFAULTS, so MONITOR_SYNTHESIS_MODEL silently
   falls back 27B -> 9B and ENABLE_MINICHECK reverts. The documented
   override-gotcha class (a stale LLM_MODEL once 404'd every generation for
   days) in its invisible variant, and a silent violation of the owner's
   never-downgrade-quality rule.
2. monitor_store.get_event_monitors: a corrupt trigger_events blob made an
   event-triggered monitor silently never fire — the "monitor quietly dead"
   class the owner keeps finding.
"""

from __future__ import annotations

import json
import logging

import pytest


class TestCorruptOverridesAreLoud:
    def test_corrupt_file_logs_error_and_boots(self, tmp_path, caplog, monkeypatch):
        """Uses the loader's own documented test seam: the module-level
        _OVERRIDES_PATH that _load_overrides resolves first."""
        import app.config as config_module
        from app.config import config

        bad = tmp_path / "config_overrides.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        monkeypatch.setattr(config_module, "_OVERRIDES_PATH", bad)
        with caplog.at_level(logging.ERROR):
            config._load_overrides()  # must not raise — boot must survive
        assert any("RUNNING ON" in r.message and "DEFAULTS" in r.message
                   for r in caplog.records), (
            "a corrupt overrides file must scream, not silently downgrade "
            "the 27B synthesis model to defaults"
        )

    def test_valid_overrides_still_apply(self, tmp_path, monkeypatch):
        import app.config as config_module
        from app.config import config

        good = tmp_path / "config_overrides.json"
        # ENABLE_MINICHECK is in _MUTABLE_FIELDS (a documented live override);
        # writing its CURRENT value keeps the test side-effect free.
        current = config.ENABLE_MINICHECK
        good.write_text(json.dumps({"ENABLE_MINICHECK": current}),
                        encoding="utf-8")
        monkeypatch.setattr(config_module, "_OVERRIDES_PATH", good)
        config._load_overrides()  # must not raise
        assert config.ENABLE_MINICHECK == current


class TestCorruptTriggerEventsAreLoud:
    def test_bad_json_row_warns_and_others_still_match(self, db, caplog):
        from app.monitors.monitor_store import MonitorStore

        store = MonitorStore(db=db)
        db.execute(
            "INSERT INTO monitors (name, check_type, check_config, enabled, "
            "trigger_events) VALUES ('Corrupt Trigger Probe', 'query', '{}', 1, "
            "'{not json')")
        db.execute(
            "INSERT INTO monitors (name, check_type, check_config, enabled, "
            "trigger_events) VALUES ('Healthy Trigger Probe', 'query', '{}', 1, "
            "'[\"internal:lesson_saved\"]')")
        with caplog.at_level(logging.WARNING):
            matched = store.get_event_monitors("internal:lesson_saved")
        names = [m.name for m in matched]
        assert "Healthy Trigger Probe" in names, \
            "one corrupt row must not stop other monitors from matching"
        assert "Corrupt Trigger Probe" not in names
        assert any("trigger_events unparseable" in r.message
                   for r in caplog.records), (
            "a monitor whose event triggers are dead must say so — silence "
            "here is a monitor that never fires and never explains why"
        )
