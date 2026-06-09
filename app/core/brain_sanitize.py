"""Final-answer sanitization — strips meta-commentary, leaked tool calls,
date-dispute disclaimers, and other artifacts before tokens reach the user.

Extracted from brain.py for size hygiene. Re-exported by brain.py so existing
`from app.core.brain import _sanitize_answer` keeps working.
"""

from __future__ import annotations

import re

# Compiled at module load — matched against every assistant message before
# it streams to the client.
_META_PATTERNS = [
    re.compile(r"\*+Note:.*?correction.*?\*+", re.IGNORECASE),
    re.compile(r"\*+Note:.*?lesson.*?\*+", re.IGNORECASE),
    re.compile(r"\*+Note:.*?(?:I'(?:ve|ll)|updated|saved|remembered|stored|recorded).*?\*+", re.IGNORECASE),
    re.compile(r"^I've (?:noted|recorded|saved|updated|stored) (?:your|that|this) correction.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Thank you for (?:the )?correction.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<think>.*</think>", re.DOTALL),
    # Defense-in-depth: strip leaked tool-call placeholders that sometimes
    # survive when the model imitates its own assistant history.
    re.compile(r"^\s*\[Calling tool:\s*[\w.\-]+\]\s*$", re.MULTILINE),
    # Qwen / Hermes-style tool call blocks — `<tool_call>{…}</tool_call>`.
    # When the extractor can't resolve the call (unknown tool name, circuit-
    # broken round, or final round emits another call instead of prose) the
    # raw block slips into the final answer. Strip both the tagged variant
    # and a bare JSON-object variant that starts with `"name":`/`"tool":`.
    re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(
        r'^\s*\{\s*"(?:name|tool|function)"\s*:\s*"[\w.\-]+"\s*,'
        r'\s*"(?:arguments|args|parameters)"\s*:\s*\{.*?\}\s*\}\s*$',
        re.DOTALL | re.MULTILINE,
    ),
    # Date confusion disclaimers (Qwen calls 2026 a "simulated future date")
    re.compile(r"\b(?:simulated|hypothetical)\s+(?:future\s+)?date\b[^.]*\.?", re.IGNORECASE),
    re.compile(r"\b(?:since|as)\s+(?:my\s+)?training\s+(?:data\s+)?cut-?off\b[^.]*\.?", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:appears?\s+to\s+be\s+)?a\s+future\s+date\b[^.]*\.?", re.IGNORECASE),
    # Stray tool-call tags that slip through when tool loop terminates w/o synthesis.
    # Existing patterns above handle the wrapped form; this catches bare orphan tags.
    re.compile(r"<tool_call>\s*$|^\s*</tool_call>", re.IGNORECASE | re.MULTILINE),
    # Meta-narration prefixes — Nova's IDENTITY block forbids these but Qwen3.5
    # baseline keeps emitting them. Strip at the line head only (so they don't
    # damage mid-sentence matches of similar phrases).
    re.compile(r"^\s*Based on (?:my|the|your) (?:research|search results?|tool results?|available (?:data|information)|knowledge(?:\s+graph)?|memory)\s*,?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*From what I (?:found|could find|gathered)\s*,?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(?:Great|Excellent|That'?s (?:a )?great|That'?s (?:an )?excellent|Fantastic|Good)\s+question!\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*I'?d be (?:happy|glad) to (?:help|assist)[^\n.]*[.!]?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*I hope (?:this|that) helps[.!]?\s*", re.IGNORECASE | re.MULTILINE),
    # Trailing "Let me know / Would you like me to" offers — strip when on
    # their own paragraph at the end of a response.
    re.compile(r"\n+\s*(?:Let me know|Feel free to (?:ask|let me know))[^\n]*$", re.IGNORECASE),
    re.compile(r"\n+\s*Would you like me to [^?\n]*\?\s*$", re.IGNORECASE),
    # "I don't have memory" false claims. Nova DOES have memory (SQLite + KG + facts).
    # Identity block forbids these; post-process strip catches weight-level slips.
    # Matches the whole sentence containing the forbidden phrase so we don't leave a
    # dangling fragment.
    re.compile(
        r"(?is)[^.!?\n]*\b(?:"
        r"I\s+don'?t\s+(?:have|retain|keep)\s+(?:(?:persistent|any|stored|old|past|previous)\s+)*(?:memory|memories)"
        r"|I\s+don'?t\s+have\s+access\s+to\s+(?:(?:our|your|previous|stored|past|any)\s+)*(?:conversations?|chats?|memory|memories)"
        r"|as\s+an\s+AI(?:\s+language\s+model)?,?\s+I\s+don'?t\s+(?:have|retain)"
        r"|I\s+don'?t\s+(?:retain|remember|recall)\s+(?:information|conversations?|anything|previous|our\s+previous)"
        r"|each\s+(?:conversation|interaction)\s+(?:with\s+me\s+)?starts?\s+fresh"
        r")\b[^.!?\n]*[.!?]?"
    ),
    # Raw tool output template headers leaking into final synthesis.
    # `memory_search` emits "## Matching User Facts" / "## Matching Conversations"
    # as structured headers so the LLM can parse — not for end-user display.
    # When weights echo them verbatim instead of synthesizing, strip them.
    re.compile(r"^\s*##\s*Matching\s+(?:User\s+Facts|Conversations|Documents)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\[Source\s+\d+:\s*\w+\]\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\[Tool\s+'[\w_]+'\s+executed\s+successfully\]\s*:?\s*", re.MULTILINE | re.IGNORECASE),
    # Raw tool-call emission when tool loop terminates without synthesis:
    # `[Tool 'monitor', "arguments": {"action": "list"}}`
    # This is an malformed variant of the tool-call JSON that slips past the
    # existing <tool_call>…</tool_call> regex. Catch the bracket-form.
    re.compile(r"\[Tool\s+'[\w_]+'\s*,\s*\"?arguments\"?\s*:\s*\{[^}]*\}+", re.IGNORECASE | re.DOTALL),
    # --- v10 boilerplate preamble that prefixes hallucinated content ---
    # "I used my tools and they returned real, live results (not simulated,
    #  not hypothetical — actual execution on the network). Today is
    #  April 21, 2026 — this is the real current date:"
    # Training-data residue. Always strip — if real tool results follow, the
    # substance stands on its own; if hallucinated content follows, stripping
    # the credibility-seeking preamble makes the lie more detectable.
    re.compile(
        r"^\s*I\s+used\s+my\s+tools\s+and\s+they\s+returned\s+real,?\s*live\s+results"
        r"[^:\n]*?(?:real\s+current\s+date\s*:|:)\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
    # --- v10 system-prompt leak framings ---
    # "The system prompt contains a directive stating..."
    # "My system prompt states..."
    # "My instructions include..."
    # "According to the system prompt given to me..."
    re.compile(
        r"(?is)[^.!?\n]*\b(?:"
        r"(?:the|my)\s+system\s+prompt\s+(?:contains?|states?|tells?|says?|directs?|instructs?)"
        r"|my\s+instructions\s+include"
        r"|according\s+to\s+the\s+system\s+prompt"
        r")\b[^.!?\n]*[.!?]?"
    ),
    # --- v10 date-dispute framings ---
    # "I cannot verify that today is April 21, 2026"
    # "I don't have access to your system clock"
    # "However, this contradicts reality"
    re.compile(
        r"(?is)[^.!?\n]*\b(?:"
        r"(?:cannot|can\'t|unable\s+to)\s+(?:independently\s+)?(?:verify|confirm)\s+(?:that\s+)?(?:today|the\s+(?:current\s+)?date|it)"
        r"|don\'?t\s+have\s+access\s+to\s+(?:your|the)\s+system\s+clock"
        r"|(?:this\s+contradicts\s+reality|cannot\s+access\s+real-time\s+news)"
        r")\b[^.!?\n]*[.!?]?"
    ),
    # --- Retrieval-label echo / self-doubt addendums ---
    # "Update: I cannot confirm the specific data points (X, Y, Z) because
    #  the retrieved context blocks [1-3] are labeled 'low relevance' with
    #  placeholder content..."
    # These came from the old critique-addendum loop AND the now-removed
    # "low relevance" inline label. Strip residue at runtime so older nova-ft
    # weights that learned the pattern don't keep emitting it.
    re.compile(
        r"(?is)\n*-{3,}\s*\n+\s*update:\s*[^.\n]*?(?:cannot\s+(?:confirm|verify)|retrieved\s+context\s+(?:block|chunk)s?\s*\[)[^\n]*"
        r"(?:\n[^\n]*?(?:placeholder|low\s+relevance|test\s+content\s+here)[^\n]*)*",
    ),
    re.compile(
        r"(?is)\b(?:retrieved\s+context\s+(?:block|chunk)s?\s*\[\d[\d\s,\-]*\]|provided\s+retrieved\s+context\s+block)\s+(?:are|is)\s+(?:labeled|labelled|marked)\s+['\"](?:low|moderate)\s+relevance['\"][^\n]*",
    ),
    re.compile(
        r"(?is)\bplaceholder\s+(?:text|content)\s+\(['\"]test\s+content\s+here['\"]\)[^\n]*",
    ),
    # --- Trailing meta-note about KG/retrieval/historical data ---
    # "*Note: The knowledge graph contains historical data from March 28
    #  showing Bitcoin at $66,000 — disregard for current pricing.*"
    # Variant of the [Note: ...] footer in non-bracket form. Strip when at
    # the end of a response (anchored to end-of-string, multi-line allowed).
    re.compile(
        r"(?is)\n+\s*\*?\*?Note\s*:?\s*[^.\n]*?(?:knowledge\s+graph|retrieved\s+context|retrieval|search\s+results?|memory\s+(?:store|search)|historical\s+data\s+from)[^\n]*\*?\*?\s*$",
    ),
    # --- "[Note: ...retrieved context...]" meta-commentary footers ---
    # Even after stripping inline relevance labels, the model sometimes appends
    # a closing bracketed note explaining what retrieval did or didn't have.
    # The user doesn't care about retrieval state — they want the answer.
    # Examples:
    #   [Note: These definitions are based on standard knowledge. The retrieved
    #    context only contained information about Merkle trees, which is unrelated.]
    #   [Note: my retrieval returned no relevant results, so this answer relies on training data]
    re.compile(
        r"(?is)\n*\[(?:note|disclaimer|caveat)\s*:[^\]]*?(?:retrieved\s+context|retrieval|search\s+results?|knowledge\s+base|memory\s+(?:store|search))[^\]]*\]\s*$",
    ),
    # --- "The previous response contained..." meta-commentary block ---
    # When the critique-driven rewrite leaks self-critique instead of
    # replacing the answer, the model writes a paragraph ABOUT its previous
    # draft. Strip it (and the "Corrected Explanation:" header that often
    # follows, plus the corrected text — because if the model went meta the
    # corrected text is usually worse than the original).
    re.compile(
        r"(?is)\n*-{3,}\s*\n+\s*(?:the\s+previous\s+response|my\s+previous\s+answer|the\s+previous\s+answer)\s+contained[^\n]*\n+(?:.*?)$",
    ),
    re.compile(
        r"(?is)\n*\*\*Corrected\s+Explanation:\*\*[^\n]*\n+(?:.*?)$",
    ),
    # --- Apologetic OPENING preamble about unrelated retrieval ---
    # "I cannot verify the specific technical claims about X from the provided
    #  sources, which only contain information about Y. Based on my general
    #  knowledge (unverified by retrieved context):"
    # Same family as the [Note: ...] footer but wrapped at the front. Strip
    # the entire opening sentence(s) until the actual answer begins. The
    # answer that follows usually starts with **Bold:**, a markdown header,
    # or a numbered/bulleted list.
    re.compile(
        r"(?is)^\s*I\s+cannot\s+(?:verify|confirm)\s+(?:the\s+)?(?:specific\s+)?(?:technical\s+)?(?:claims?|facts?|details?|information)?[^\n]*?"
        r"(?:provided\s+sources|retrieved\s+context|knowledge\s+base|search\s+results)[^\n]*?\n+"
        r"(?:Based\s+on\s+my\s+(?:general|prior|training|own)\s+knowledge[^\n]*\n+)?",
    ),
    # --- "I cannot and will not" prompt-injection refusal boilerplate ---
    # When the model resists prompt injection it sometimes lapses into
    # corporate-AI tone ("designed to follow security boundaries", "system-
    # level operation outside my capabilities"). Sovereign Nova should refuse
    # in plain language ("no, that won't work because X"), not preachy boilerplate.
    re.compile(
        r"(?is)(?:^|\n)\s*I\s+cannot\s+and\s+will\s+not[^.\n]*\.\s*(?:This\s+appears\s+to\s+be\s+a\s+privilege\s+escalation\s+attempt[^.\n]*\.\s*)?(?:I\'?m\s+designed\s+to\s+follow\s+security\s+boundaries[^.\n]*\.\s*)?",
    ),
    re.compile(
        r"(?is)\bI\s+(?:also\s+)?don\'?t\s+(?:repeat|execute)\s+instructions\s+from\s+prompts\s+that\s+try\s+to\s+override\s+my\s+system\s+instructions[^.\n]*\.\s*",
    ),
]


def _truncate_repetition_loop(text: str) -> str:
    """Detect when the model fell into a paragraph-repetition loop and
    truncate at the first repetition. Compares 200-char paragraphs; if the
    same paragraph appears 3+ times, keep only the first occurrence and
    append a single short note. This catches the Qwen3.5 LaTeX-loop failure
    mode (verified 2026-05-04 IEEE 754 query).
    """
    if not text or len(text) < 800:
        return text
    # Split on double-newline paragraph boundaries
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 4:
        return text
    # Detect repetition: hash each para's first 200 chars; if any hash
    # appears 3+ times, find the first repeat and truncate there.
    seen: dict[str, int] = {}
    cutoff_idx: int | None = None
    for i, p in enumerate(paragraphs):
        key = p.strip()[:200]
        if not key or len(key) < 60:
            continue  # ignore very short paragraphs (headers, separators)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3 and cutoff_idx is None:
            # Find the first occurrence of this paragraph
            for j, q in enumerate(paragraphs[:i]):
                if q.strip()[:200] == key:
                    cutoff_idx = j + 1  # keep through first occurrence
                    break
            break
    if cutoff_idx is None:
        return text
    truncated = "\n\n".join(paragraphs[:cutoff_idx])
    return truncated.rstrip() + "\n\n[...]"


def _sanitize_answer(text: str) -> str:
    """Strip meta-commentary and internal markers from final answers.

    Returns empty when stripping removes all content — callers are
    responsible for handling the empty case appropriately. Monitor paths
    should let empty pass through (so downstream formatters can decide
    whether to alert); interactive chat paths should substitute a polite
    fallback at the /api/chat layer, not inside the streaming path.
    """
    if not isinstance(text, str):
        return str(text) if text else ""
    for pat in _META_PATTERNS:
        text = pat.sub("", text)
    # Collapse excess blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    # Catch generation degenerates that survived repeat_penalty
    text = _truncate_repetition_loop(text)
    return text.strip()
