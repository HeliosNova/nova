"""One-off refactor: split domain_overview into _gather_evidence + _synthesize_from_evidence
so the ceiling A/B harness can replay the EXACT production synthesis on captured evidence.
Marker-based (robust to line shifts). Behavior-preserving move + one syn_model-param tweak.
"""
import io

P = r"F:\Helios Project\nova_\app\monitors\deep_research.py"
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")


def find(pred, lo=0, hi=None):
    hi = len(lines) if hi is None else hi
    for i in range(lo, hi):
        if pred(lines[i]):
            return i
    raise SystemExit(f"marker not found (lo={lo})")


def_i = find(lambda l: l.startswith("async def domain_overview("))
ds_open = find(lambda l: l.strip().startswith('"""'), def_i, def_i + 3)
ds_close = find(lambda l: l.rstrip().endswith('"""'), ds_open + 1)
today_i = find(lambda l: l.strip().startswith("today = _NOW()"), ds_close + 1)
gather_i = find(lambda l: l.strip().startswith("subjects = await _focus_subjects("), today_i)
fallback_i = find(lambda l: l.strip() == "if len(findings) < 2:", gather_i)
synth_i = find(lambda l: l.strip().startswith("hosts = sorted({_host("), fallback_i)
end_i = find(lambda l: l.strip() == "return header + final", synth_i)

# --- gather block: gather_i .. before fallback (trim trailing blanks) ---
gb_end = fallback_i - 1
while lines[gb_end].strip() == "":
    gb_end -= 1
gather_block = lines[gather_i:gb_end + 1]

# --- fallback block (stays in domain_overview): fallback_i .. before synth (trim blanks) ---
fb_end = synth_i - 1
while lines[fb_end].strip() == "":
    fb_end -= 1
fallback_block = lines[fallback_i:fb_end + 1]

# --- synthesis block: synth_i .. end_i, with the syn_model line made param-aware ---
synth_text = "\n".join(lines[synth_i:end_i + 1])
_old = '    syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None'
_new = ('    if syn_model is None:\n'
        '        syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None')
assert _old in synth_text, "syn_model line not found in synthesis block"
synth_text = synth_text.replace(_old, _new, 1)

docstring = "\n".join(lines[def_i:ds_close + 1])  # def line + full docstring

gather_fn = (
    "async def _gather_evidence(label: str, n_stories: int, feed_key: str | None):\n"
    '    """Story selection + pooled deep read + iterative gap loop -> (subjects, findings,\n'
    "    articles). Split out of domain_overview (2026-07-10) so the ceiling A/B harness can\n"
    '    capture evidence ONCE and replay synthesis on it. Behavior-preserving move."""\n'
    + "\n".join(gather_block) + "\n"
    "    return subjects, findings, articles\n"
)
synth_fn = (
    "async def _synthesize_from_evidence(label: str, findings: list, articles: list, today: str,"
    " *, kg=None, syn_model: str | None = None) -> str:\n"
    '    """Deep analysis -> best-of-N synthesis -> grounding stack -> fact-banking -> bound.\n'
    "    Split out of domain_overview (2026-07-10) so the ceiling harness replays the EXACT\n"
    "    production synthesis on captured evidence with any (model, config). syn_model=None\n"
    '    -> config.MONITOR_SYNTHESIS_MODEL."""\n'
    + synth_text + "\n"
)
new_domain = (
    docstring + "\n"
    '    today = _NOW().strftime("%B %d, %Y")\n'
    "    subjects, findings, articles = await _gather_evidence(label, n_stories, feed_key)\n"
    + "\n".join(fallback_block) + "\n"
    "    return await _synthesize_from_evidence(label, findings, articles, today, kg=kg)\n"
)

replacement = gather_fn + "\n\n" + synth_fn + "\n\n" + new_domain
new_lines = lines[:def_i] + replacement.split("\n") + lines[end_i + 1:]
out = "\n".join(new_lines)

# sanity: compile
import ast
ast.parse(out)
io.open(P, "w", encoding="utf-8").write(out)
print(f"OK: refactored. def_i={def_i} gather={gather_i} fallback={fallback_i} synth={synth_i} end={end_i}")
print(f"gather_block={len(gather_block)} lines, fallback={len(fallback_block)} lines, synth={end_i-synth_i+1} lines")
