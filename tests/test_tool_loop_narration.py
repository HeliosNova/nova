"""A tool-backed answer that narrates its next move is not finished (2026-09-09).

Nightly eval, knowing_dossier_paraphrase, twice on 2026-09-09: after five web
rounds to the circuit breaker, a deep-research call and a browser 404, the
model's final text was

    The direct link returned a 404 error (the page may be moved or the URL
    incorrect). Let me try navigating to fusionreview.com's homepage and
    search from there for "Aurora-7 Iceland" content, or use web_search with
    more specific terms like "compact tokamak Iceland July 2026".

and that was graded as the answer. The unfinished-draft repair already exists
for exactly this shape, but its detector only looked at the last 200
characters for "let me <verb>" ending the text, and this narration runs on
past its verb. When tools ran, a final text that announces a next move
ANYWHERE is process narration, not an answer, and the tool-less repair pass
gets to write the real one from what was gathered.
"""
from __future__ import annotations

import app.core.brain as brain

LIVE = ('The direct link returned a 404 error (the page may be moved or the URL incorrect). '
        'Let me try navigating to fusionreview.com\'s homepage and search from there for '
        '"Aurora-7 Iceland" content, or use web_search with more specific terms like '
        '"compact tokamak Iceland July 2026".')

FINISHED = ("The Aurora-7 pilot plant in Iceland sustained a fusion gain of Q=2.1 for 43 seconds "
            "in July 2026 (fusionreview.com). Whether Q>3 is reachable without a divertor "
            "redesign is still open.")


def test_the_live_narration_is_unfinished_when_tools_ran():
    assert brain._draft_is_unfinished(LIVE, tools_ran=True)


def test_the_same_text_is_left_alone_when_no_tools_ran():
    """Chat without tools keeps the old, narrow rule."""
    assert not brain._draft_is_unfinished(LIVE)


def test_a_finished_tool_backed_answer_is_finished():
    assert not brain._draft_is_unfinished(FINISHED, tools_ran=True)


def test_the_old_tail_rule_still_holds():
    assert brain._draft_is_unfinished("Here is what I have so far. Let me fetch the rest:")
