"""Delivery gate for stat-line canary monitors (2026-08-26).

Generic numeric change-detection re-delivered healthy readouts every cycle
(the numbers always drift ≥5%): "kg growth normal" 3x/day and "✅ ollama
healthy (6ms)" on every latency jitter, both stored as status=alert. The
gate suppresses healthy→healthy and keeps warnings + the recovery edge loud.
"""
from app.monitors.heartbeat_loop import _canary_should_alert

NORMAL_A = "📊 kg growth normal (+0.8%) │ last_6h: 130 │ prev_6h: 129"
NORMAL_B = "📊 kg growth normal (-3.2%) │ last_6h: 120 │ prev_6h: 124"
SPIKE = "⚠️ kg growth spike (+66.3% over prev 6h) │ last_6h: 168 │ prev_6h: 101"
FLATLINE = "⚠️ KG extraction FLATLINE — 77 digests ran in 24h but 0 facts extracted"

OLLAMA_OK_A = "✅ ollama healthy (6ms) │ latency: 6ms"
OLLAMA_OK_B = "✅ ollama healthy (2ms) │ latency: 2ms"
OLLAMA_SLOW = "⚠️ ollama slow (2400ms) │ latency: 2400ms"


class TestKgGrowthGate:
    def test_normal_to_normal_suppressed(self):
        assert _canary_should_alert("kg_growth", NORMAL_A, NORMAL_B) is False

    def test_normal_to_spike_delivers(self):
        assert _canary_should_alert("kg_growth", NORMAL_A, SPIKE) is True

    def test_spike_repeats(self):
        assert _canary_should_alert("kg_growth", SPIKE, SPIKE) is True

    def test_recovery_edge_delivers_once(self):
        assert _canary_should_alert("kg_growth", SPIKE, NORMAL_A) is True

    def test_flatline_delivers(self):
        assert _canary_should_alert("kg_growth", NORMAL_A, FLATLINE) is True

    def test_missing_last_result_delivers(self):
        assert _canary_should_alert("kg_growth", None, NORMAL_A) is True


class TestOllamaLatencyGate:
    def test_healthy_jitter_suppressed(self):
        assert _canary_should_alert("ollama_latency", OLLAMA_OK_A, OLLAMA_OK_B) is False

    def test_slow_delivers(self):
        assert _canary_should_alert("ollama_latency", OLLAMA_OK_A, OLLAMA_SLOW) is True

    def test_recovery_delivers(self):
        assert _canary_should_alert("ollama_latency", OLLAMA_SLOW, OLLAMA_OK_A) is True


class TestUnmappedCheckTypesUnaffected:
    def test_query_monitors_always_pass_through(self):
        assert _canary_should_alert("query", "same text", "same text") is True
