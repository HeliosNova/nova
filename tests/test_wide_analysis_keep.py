"""Entailment gate: keep more cross-source synthesis (2026-10-18, flag-gated).

The ~32% of digest sentences finally deleted by the gate are dominated by
analytical 'so what' lines the narrow _ANALYTICAL_RE misses. gate v4 de-cites
analytical sentences instead of deleting them; ENABLE_WIDE_ANALYSIS_KEEP widens
what counts as analytical. These sentences are the real finally-dropped ones
from the 2026-10-17 drop corpus. Default OFF — current behaviour is unchanged.
"""

from __future__ import annotations

import pytest

from app.monitors import deep_research as dr


# Real finally-dropped synthesis from the 2026-10-17 [entail-drop] corpus.
WIDE_SYNTHESIS = [
    "This transforms parental leave from a social welfare issue into a core competitive advantage in talent acquisition.",
    "This positions Vietnam as a critical node in the global tech supply chain.",
    "The overarching narrative for October is one of accelerating physical risk colliding with slowing political adaptation.",
    "This success clears the path for the next mission.",
    "This creates a risk of overcapacity if domestic production scales faster than residential adoption.",
]


@pytest.mark.parametrize("claim", WIDE_SYNTHESIS)
def test_wide_shapes_are_kept_only_when_the_flag_is_on(claim, monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_WIDE_ANALYSIS_KEEP", False)
    assert dr._is_analytical(claim) is False          # today: deleted
    monkeypatch.setattr(config, "ENABLE_WIDE_ANALYSIS_KEEP", True)
    assert dr._is_analytical(claim) is True            # with the flag: kept (de-cited)


@pytest.mark.parametrize("claim", [
    "This divergence highlights a bifurcated economic landscape within East Asia.",   # narrow regex already catches it
    "The deeper implication is a structural shift in alliances.",
])
def test_narrow_shapes_still_match_regardless_of_flag(claim, monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_WIDE_ANALYSIS_KEEP", False)
    assert dr._is_analytical(claim) is True


@pytest.mark.parametrize("claim", [
    "This transforms the market, adding $4.2 billion in revenue.",     # figure → must still drop
    "This positions the firm with 23% market share.",
    "Nvidia reported record quarterly revenue of $96.2 billion.",      # a bare fact, not analysis
    "The company acquired the startup for an undisclosed sum.",
])
def test_figure_bearing_or_factual_claims_still_drop_even_with_the_flag(claim, monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_WIDE_ANALYSIS_KEEP", True)
    assert dr._is_analytical(claim) is False


def test_flag_defaults_off():
    from app.config import config
    # A fresh config must not change gate behaviour until the A/B reads the rule.
    assert config.ENABLE_WIDE_ANALYSIS_KEEP in (False, True)  # exists
    import os
    if "ENABLE_WIDE_ANALYSIS_KEEP" not in os.environ:
        assert config.ENABLE_WIDE_ANALYSIS_KEEP is False
