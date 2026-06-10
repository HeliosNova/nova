"""Nova configuration — loaded from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning("Invalid integer for %s='%s', using default %d", key, val, default)
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning("Invalid float for %s='%s', using default %s", key, val, default)
        return default


_OVERRIDES_PATH = Path(os.getenv("CONFIG_OVERRIDES_PATH", "/data/config_overrides.json"))

# Fields that may be persisted/loaded via config overrides (security: prevents persisted bypasses)
_MUTABLE_FIELDS = {
    "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_URL",
    "VISION_MODEL",
    "EMBEDDING_MODEL", "RETRIEVAL_TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "MAX_HISTORY_MESSAGES", "MAX_LESSONS_IN_PROMPT", "MAX_SKILLS_CHECK",
    "MAX_CONTEXT_TOKENS", "RECENT_MESSAGES_KEEP",
    "CODE_EXEC_TIMEOUT", "MAX_TOOL_ROUNDS", "MAX_SAME_TOOL_CALLS", "MAX_TOOL_CALLS_PER_QUERY",
    "SHELL_EXEC_TIMEOUT",
    "BROWSER_TIMEOUT", "TOOL_TIMEOUT", "GENERATION_TIMEOUT", "INTERNAL_LLM_TIMEOUT",
    "ENABLE_PLANNING", "ENABLE_CRITIQUE", "ENABLE_CUSTOM_TOOLS",
    "ENABLE_EXTENDED_THINKING", "ENABLE_DELEGATION", "ENABLE_CURIOSITY",
    "ENABLE_VOICE", "ENABLE_TTS", "TTS_MODEL_PATH", "ENABLE_MODEL_ROUTING",
    "ENABLE_HEARTBEAT", "HEARTBEAT_INTERVAL", "ENABLE_PROACTIVE",
    "ENABLE_SHELL_EXEC", "ENABLE_MCP", "ENABLE_MCP_SERVER",
    "ENABLE_AUTO_SKILL_CREATION", "ENABLE_AUTONOMOUS_TOOL_CREATION",
    "AUTO_TOOL_CREATION_THRESHOLD", "ENABLE_INJECTION_DETECTION",
    "ENABLE_DESKTOP_AUTOMATION", "ENABLE_WEBHOOKS", "ENABLE_EMAIL_SEND",
    "ENABLE_INTEGRATIONS", "ENABLE_CALENDAR",
    "MIN_MONITOR_SCHEDULE_SECONDS", "EMAIL_RATE_LIMIT",
    "WEB_SEARCH_TIMEOUT", "WEB_SEARCH_ENGINES", "WEB_SEARCH_MAX_RESULTS",
    "ALLOWED_ORIGINS",
    "MAX_SYSTEM_TOKENS", "MAX_USER_FACTS", "MAX_KG_FACTS", "MAX_LESSON_CANDIDATES",
    "MAX_CURIOSITY_PENDING", "MAX_CURIOSITY_ATTEMPTS", "MAX_CURIOSITY_QUEUE_SIZE",
    "MAX_CUSTOM_TOOL_CODE_LENGTH", "MAX_CUSTOM_TOOLS", "RATE_LIMIT_RPM",
    "MAX_KG_FACTS_IN_PROMPT", "MAX_REFLEXIONS_IN_PROMPT", "MAX_SUCCESS_PATTERNS_IN_PROMPT",
    "MAX_REFLEXIONS",
    "CRITIQUE_ANSWER_LIMIT", "CRITIQUE_SOURCES_LIMIT", "CRITIQUE_FACTS_LIMIT",
    "MAX_CRITIQUE_ROUNDS", "DIGEST_HOUR", "USER_TIMEZONE",
    # Tuning parameters
    "RESPONSE_TOKEN_BUDGET", "RETRIEVAL_RELEVANCE_THRESHOLD",
    "TEMPERATURE_DEFAULT",
    "MIN_RRF_SCORE", "LESSON_VECTOR_MAX_DISTANCE", "KG_VECTOR_MAX_DISTANCE", "DEDUP_JACCARD_THRESHOLD",
    "REFLEXION_DECAY_DAYS", "REFLEXION_DECAY_AMOUNT", "REFLEXION_DISTANCE_THRESHOLD",
    "ENABLE_SEMANTIC_SKILL_MATCHING", "SKILL_SEMANTIC_THRESHOLD",
    "SKILL_EMA_ALPHA", "SKILL_STALE_DAYS",
    "INJECTION_SUSPICIOUS_THRESHOLD",
    "REFLEXION_FAILURE_THRESHOLD", "REFLEXION_SUCCESS_THRESHOLD",
    "KG_GRAPH_MAX_FRONTIER", "AUTH_MAX_TRACKED_IPS",
    "ENABLE_EVAL_HARNESS", "EVAL_SUITE_PATH", "EVAL_REPORT_PATH", "EVAL_REGRESSION_TOLERANCE",
    "ENABLE_MULTI_AGENT", "MULTI_AGENT_TRIGGER_THRESHOLD", "MAX_AGENT_COUNT", "AGENT_TASK_TIMEOUT", "MAX_PARALLEL_AGENTS", "MAX_STRUCTURAL_DEPTH",
    "ENABLE_TREE_OF_THOUGHT", "TOT_SAMPLE_N",
    "ENABLE_BEST_OF_N", "BEST_OF_N_SAMPLES", "BEST_OF_N_QUALITY_THRESHOLD",
    "RETRIEVAL_HARD_FLOOR",
    "ENABLE_RERANKER", "RETRIEVAL_RRF_K",
    "ENABLE_PPR_RETRIEVAL", "ENABLE_CONFORMAL_ABSTENTION", "ENABLE_GSW_EPISODIC",
    "ENABLE_LORA_CONTINUAL_MERGE", "LORA_MERGE_ALPHA", "ENABLE_SFT_BOOTSTRAP",
    "ENABLE_RLVR_SIGNALS", "ENABLE_DEBATE", "DEBATE_TIMEOUT_SECONDS",
    "ENABLE_PROCEDURAL_CONSOLIDATION",
    "ENABLE_MAD_MM_MASKING", "MAD_MM_MIN_PRIOR_STEPS", "MAD_MM_JUDGE_MODEL",
    "ENABLE_TWO_PHASE_DREAM", "DREAM_REM_TIMEOUT_SECONDS",
    # Prompt self-modification
    "ENABLE_PROMPT_SELF_MOD",
    "PROMPT_MOD_MAX_PROPOSALS_PER_DAY", "PROMPT_MOD_MAX_PENDING",
    "PROMPT_MOD_MAX_PROMOTIONS_PER_DAY", "PROMPT_MOD_MAX_DRIFT",
    "PROMPT_MOD_MIN_IMPROVEMENT_PP", "PROMPT_MOD_REGRESSION_TOLERANCE_PP",
    "PROMPT_MOD_STABILITY_RUNS", "PROMPT_MOD_LATENCY_OVERHEAD_MAX",
}


@dataclass
class Config:
    # LLM — Nova is Ollama-only (cloud providers removed for sovereign operation).
    LLM_PROVIDER: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    LLM_MODEL: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5:27b"))
    OLLAMA_URL: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://ollama:11434"))

    # Learning / reflexion toggles — these were referenced in code but missing
    # from config, leading to `AttributeError: ??? ` at inspection time.
    ENABLE_REFLEXION: bool = field(default_factory=lambda: _env("ENABLE_REFLEXION", "true").lower() == "true")
    ENABLE_BACKGROUND_TASKS: bool = field(default_factory=lambda: _env("ENABLE_BACKGROUND_TASKS", "true").lower() == "true")
    ENABLE_AUTO_FINETUNE: bool = field(default_factory=lambda: _env("ENABLE_AUTO_FINETUNE", "false").lower() == "true")

    # MCP (Model Context Protocol) — client (consume external MCP tools)
    ENABLE_MCP: bool = field(default_factory=lambda: _env("ENABLE_MCP", "true").lower() == "true")
    MCP_CONFIG_DIR: str = field(default_factory=lambda: _env("MCP_CONFIG_DIR", "/data/mcp"))

    # MCP Server (expose Nova as MCP server)
    ENABLE_MCP_SERVER: bool = field(default_factory=lambda: _env("ENABLE_MCP_SERVER", "true").lower() == "true")
    MCP_SERVER_NAME: str = field(default_factory=lambda: _env("MCP_SERVER_NAME", "nova"))

    # External skills (AgentSkills format)
    SKILLS_DIR: str = field(default_factory=lambda: _env("SKILLS_DIR", "/data/skills"))

    # Memory
    MAX_HISTORY_MESSAGES: int = field(default_factory=lambda: _env_int("MAX_HISTORY_MESSAGES", 50))
    MAX_LESSONS_IN_PROMPT: int = field(default_factory=lambda: _env_int("MAX_LESSONS_IN_PROMPT", 5))
    MAX_SKILLS_CHECK: int = field(default_factory=lambda: _env_int("MAX_SKILLS_CHECK", 500))

    # Context window management
    MAX_CONTEXT_TOKENS: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 16000))
    RECENT_MESSAGES_KEEP: int = field(default_factory=lambda: _env_int("RECENT_MESSAGES_KEEP", 12))

    # Retrieval
    # Embedder for all ChromaDB collections. Resolved by app/core/embedding.py:
    # a reachable Ollama embedder is used; otherwise falls back to ChromaDB's
    # bundled all-MiniLM-L6-v2. bge-m3 won a 2026 paraphrase-retrieval bake-off
    # (r@3=1.00 vs MiniLM 0.95); see embedding.py. Set to "default" to force
    # MiniLM (zero GPU/network cost) on modest hardware.
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "bge-m3"))
    RETRIEVAL_TOP_K: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    CHUNK_SIZE: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    RRF_K: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    RETRIEVAL_RRF_K: int = field(default_factory=lambda: _env_int("RETRIEVAL_RRF_K", 60))
    ENABLE_RERANKER: bool = field(default_factory=lambda: _env("ENABLE_RERANKER", "true").lower() == "true")
    # HippoRAG 2 PPR-over-KG retrieval (graph walk for multi-hop fact recall)
    ENABLE_PPR_RETRIEVAL: bool = field(default_factory=lambda: _env("ENABLE_PPR_RETRIEVAL", "true").lower() == "true")
    # Conformal Abstention — calibrated confidence-footer thresholds
    ENABLE_CONFORMAL_ABSTENTION: bool = field(default_factory=lambda: _env("ENABLE_CONFORMAL_ABSTENTION", "true").lower() == "true")
    # GSW (Generative Semantic Workspace) — episodic memory layer
    ENABLE_GSW_EPISODIC: bool = field(default_factory=lambda: _env("ENABLE_GSW_EPISODIC", "true").lower() == "true")
    # Continual LoRA merging (TIES) — preserve prior adapter knowledge across fine-tunes
    ENABLE_LORA_CONTINUAL_MERGE: bool = field(default_factory=lambda: _env("ENABLE_LORA_CONTINUAL_MERGE", "false").lower() == "true")
    LORA_MERGE_ALPHA: float = field(default_factory=lambda: _env_float("LORA_MERGE_ALPHA", 0.5))
    # open-rs SFT pre-DPO bootstrap — short SFT epoch on reasoning traces before DPO
    ENABLE_SFT_BOOTSTRAP: bool = field(default_factory=lambda: _env("ENABLE_SFT_BOOTSTRAP", "false").lower() == "true")
    # RLVR — record verifiable signals (tool/JSON/math/claim/quiz/code outcomes) so a
    # later GRPO/RLVR fine-tune can train on real rewards instead of LLM-judge noise.
    ENABLE_RLVR_SIGNALS: bool = field(default_factory=lambda: _env("ENABLE_RLVR_SIGNALS", "true").lower() == "true")
    # A-HMAD style debate — role-specialized critics + judge on high-stakes drafts.
    # Off by default: latency cost is ~3 extra LLM calls; opt in per environment.
    ENABLE_DEBATE: bool = field(default_factory=lambda: _env("ENABLE_DEBATE", "false").lower() == "true")
    DEBATE_TIMEOUT_SECONDS: float = field(default_factory=lambda: _env_float("DEBATE_TIMEOUT_SECONDS", 240.0))
    # MAD-MM (ICLR 2026) subjective memory masking in the agent_loop iteration.
    # When a step is retried (attempts > 0) AND there are >= MAD_MM_MIN_PRIOR_STEPS
    # done steps to draw on, ask the LLM which prior step observations and findings
    # are useful for the current step before re-rendering the scratchpad. The
    # original attempt apparently got misled — masking drops the most likely
    # culprits. Opt-in (default off) because it adds 1 batched LLM call per retry.
    ENABLE_MAD_MM_MASKING: bool = field(default_factory=lambda: _env("ENABLE_MAD_MM_MASKING", "false").lower() == "true")
    MAD_MM_MIN_PRIOR_STEPS: int = field(default_factory=lambda: _env_int("MAD_MM_MIN_PRIOR_STEPS", 3))
    # Judge model for MAD-MM mask LLM calls. A stronger third-party model
    # discriminates better than the production fine-tune (which is conservative
    # and tends to keep everything). Empty string = use the default LLM_MODEL.
    MAD_MM_JUDGE_MODEL: str = field(default_factory=lambda: _env("MAD_MM_JUDGE_MODEL", "qwen3.6:27b"))
    # SCM/SleepGate-style two-phase dream consolidation: split the current
    # kitchen-sink Phase 3 into NREM (structural ops: prune/compact/disable —
    # fast, deterministic) and REM (integrative ops: promote/resolve/distill —
    # slow, LLM-driven). Failures in REM no longer roll back NREM. Opt-in
    # (default off) for prototype; controlled rollout. See `consolidate_nrem` +
    # `consolidate_rem` in `app/core/dream.py`.
    ENABLE_TWO_PHASE_DREAM: bool = field(default_factory=lambda: _env("ENABLE_TWO_PHASE_DREAM", "false").lower() == "true")
    DREAM_REM_TIMEOUT_SECONDS: float = field(default_factory=lambda: _env_float("DREAM_REM_TIMEOUT_SECONDS", 60.0))
    # Procedural memory consolidation in dream — cluster near-duplicate lessons,
    # generalize via LLM, demote subsumed members so retrieval prefers the canonical.
    ENABLE_PROCEDURAL_CONSOLIDATION: bool = field(default_factory=lambda: _env("ENABLE_PROCEDURAL_CONSOLIDATION", "true").lower() == "true")

    # Tools
    SEARXNG_URL: str = field(default_factory=lambda: _env("SEARXNG_URL", "http://searxng:8080"))
    # Bumped 20s → 35s after eval-suite analysis (2026-05-09): SearXNG hits
    # multiple upstream engines, slow ones flake the whole call. User mandate:
    # optimize for best, not fastest.
    WEB_SEARCH_TIMEOUT: float = field(default_factory=lambda: _env_float("WEB_SEARCH_TIMEOUT", 35.0))
    WEB_SEARCH_ENGINES: str = field(default_factory=lambda: _env("WEB_SEARCH_ENGINES", "bing,startpage,ecosia,yandex,yahoo"))
    WEB_SEARCH_MAX_RESULTS: int = field(default_factory=lambda: _env_int("WEB_SEARCH_MAX_RESULTS", 5))
    CODE_EXEC_TIMEOUT: int = field(default_factory=lambda: _env_int("CODE_EXEC_TIMEOUT", 15))
    # Worst-case bound on agentic tool-use rounds per query. A 2026 latency/depth
    # study (7-query mix, cap 10 vs 5) found chat uses <=3 rounds (even a fictional
    # "find the codename" query self-limited to 3), so 10 was dead headroom and the
    # cap is NOT a latency lever (latency is dominated by per-round 9B+browser cost,
    # not round count). 6 keeps comfortable headroom for chat + monitor research
    # while halving the worst-case spin ceiling. Insurance, not an optimization.
    MAX_TOOL_ROUNDS: int = field(default_factory=lambda: _env_int("MAX_TOOL_ROUNDS", 6))
    MAX_SAME_TOOL_CALLS: int = field(default_factory=lambda: _env_int("MAX_SAME_TOOL_CALLS", 3))
    MAX_TOOL_CALLS_PER_QUERY: int = field(default_factory=lambda: _env_int("MAX_TOOL_CALLS_PER_QUERY", 15))
    SHELL_EXEC_TIMEOUT: int = field(default_factory=lambda: _env_int("SHELL_EXEC_TIMEOUT", 45))
    BROWSER_TIMEOUT: int = field(default_factory=lambda: _env_int("BROWSER_TIMEOUT", 60))
    BROWSER_CDP_URL: str = field(default_factory=lambda: _env("BROWSER_CDP_URL", ""))  # http:// CDP URL to connect to host browser
    TOOL_TIMEOUT: int = field(default_factory=lambda: _env_int("TOOL_TIMEOUT", 180))
    TOOL_OUTPUT_MAX_CHARS: int = field(default_factory=lambda: _env_int("TOOL_OUTPUT_MAX_CHARS", 10000))
    GENERATION_TIMEOUT: int = field(default_factory=lambda: _env_int("GENERATION_TIMEOUT", 900))
    INTERNAL_LLM_TIMEOUT: int = field(default_factory=lambda: _env_int("INTERNAL_LLM_TIMEOUT", 60))
    ENABLE_SHELL_EXEC: bool = field(default_factory=lambda: _env("ENABLE_SHELL_EXEC", "false").lower() == "true")

    # Desktop automation (requires display server + PyAutoGUI)
    ENABLE_DESKTOP_AUTOMATION: bool = field(default_factory=lambda: _env("ENABLE_DESKTOP_AUTOMATION", "false").lower() == "true")
    DESKTOP_CLICK_DELAY: float = field(default_factory=lambda: _env_float("DESKTOP_CLICK_DELAY", 0.5))
    SCREENSHOT_DIR: str = field(default_factory=lambda: _env("SCREENSHOT_DIR", "/tmp/nova_screenshots"))

    # Heartbeat / Proactive
    ENABLE_HEARTBEAT: bool = field(default_factory=lambda: _env("ENABLE_HEARTBEAT", "true").lower() == "true")
    HEARTBEAT_INTERVAL: int = field(default_factory=lambda: _env_int("HEARTBEAT_INTERVAL", 60))
    ENABLE_PROACTIVE: bool = field(default_factory=lambda: _env("ENABLE_PROACTIVE", "true").lower() == "true")
    ENABLE_EVENT_TRIGGERS: bool = field(default_factory=lambda: _env("ENABLE_EVENT_TRIGGERS", "false").lower() == "true")
    MIN_MONITOR_SCHEDULE_SECONDS: int = field(default_factory=lambda: _env_int("MIN_MONITOR_SCHEDULE_SECONDS", 60))
    DIGEST_HOUR: int = field(default_factory=lambda: _env_int("DIGEST_HOUR", 21))
    USER_TIMEZONE: str = field(default_factory=lambda: _env("USER_TIMEZONE", "UTC"))

    # Automated eval harness
    ENABLE_EVAL_HARNESS: bool = field(default_factory=lambda: _env("ENABLE_EVAL_HARNESS", "true").lower() == "true")
    EVAL_SUITE_PATH: str = field(default_factory=lambda: _env("EVAL_SUITE_PATH", "evals/suite.yaml"))
    EVAL_REPORT_PATH: str = field(default_factory=lambda: _env("EVAL_REPORT_PATH", "/data/eval_reports"))
    EVAL_REGRESSION_TOLERANCE: float = field(default_factory=lambda: _env_float("EVAL_REGRESSION_TOLERANCE", 0.10))

    # Learning
    TRAINING_DATA_PATH: str = field(default_factory=lambda: _env("TRAINING_DATA_PATH", "/data/training_data.jsonl"))
    MAX_TRAINING_PAIRS: int = field(default_factory=lambda: _env_int("MAX_TRAINING_PAIRS", 10000))
    MAX_LESSONS: int = field(default_factory=lambda: _env_int("MAX_LESSONS", 500))
    TRAINING_DATA_CHANNELS: str = field(default_factory=lambda: _env("TRAINING_DATA_CHANNELS", "api"))  # comma-separated: api,discord,telegram,whatsapp,signal

    # Fine-tuning automation
    # FINETUNE_MIN_NEW_PAIRS raised 15→100 on 2026-05-14 (task #23) to match
    # the monthly cadence policy: at ~3.3 pairs/day organic accumulation,
    # 100 new pairs = ~30 days. v16 was trained on 699 pairs, so 100 is
    # ~14% delta — enough to A/B meaningfully against the deployed model.
    # The earlier 15-pair threshold notified after ~4 days, well below the
    # noise floor (the #41 smoke run on 19 GRPO-derived pairs A/B'd 10/10
    # ties — proving very-small deltas don't differentiate).
    FINETUNE_MIN_NEW_PAIRS: int = field(default_factory=lambda: _env_int("FINETUNE_MIN_NEW_PAIRS", 100))
    FINETUNE_OUTPUT_DIR: str = field(default_factory=lambda: _env("FINETUNE_OUTPUT_DIR", "/data/finetune"))

    # Reasoning
    ENABLE_PLANNING: bool = field(default_factory=lambda: _env("ENABLE_PLANNING", "true").lower() == "true")
    ENABLE_CRITIQUE: bool = field(default_factory=lambda: _env("ENABLE_CRITIQUE", "true").lower() == "true")
    ENABLE_CUSTOM_TOOLS: bool = field(default_factory=lambda: _env("ENABLE_CUSTOM_TOOLS", "true").lower() == "true")
    ENABLE_EXTENDED_THINKING: bool = field(default_factory=lambda: _env("ENABLE_EXTENDED_THINKING", "true").lower() == "true")

    # Model routing
    VISION_MODEL: str = field(default_factory=lambda: _env("VISION_MODEL", "qwen3.5:9b"))
    FAST_MODEL: str = field(default_factory=lambda: _env("FAST_MODEL", "qwen3.5:4b"))
    HEAVY_MODEL: str = field(default_factory=lambda: _env("HEAVY_MODEL", ""))
    ENABLE_MODEL_ROUTING: bool = field(default_factory=lambda: _env("ENABLE_MODEL_ROUTING", "true").lower() == "true")

    # Critique
    MAX_CRITIQUE_ROUNDS: int = field(default_factory=lambda: _env_int("MAX_CRITIQUE_ROUNDS", 3))
    CRITIQUE_ANSWER_LIMIT: int = field(default_factory=lambda: _env_int("CRITIQUE_ANSWER_LIMIT", 1500))
    CRITIQUE_SOURCES_LIMIT: int = field(default_factory=lambda: _env_int("CRITIQUE_SOURCES_LIMIT", 1500))
    CRITIQUE_FACTS_LIMIT: int = field(default_factory=lambda: _env_int("CRITIQUE_FACTS_LIMIT", 2000))

    # Delegation (LLM-driven, via DelegateTool)
    ENABLE_DELEGATION: bool = field(default_factory=lambda: _env("ENABLE_DELEGATION", "true").lower() == "true")
    MAX_DELEGATION_DEPTH: int = field(default_factory=lambda: _env_int("MAX_DELEGATION_DEPTH", 1))

    # Multi-agent structural decomposition
    ENABLE_MULTI_AGENT: bool = field(default_factory=lambda: _env("ENABLE_MULTI_AGENT", "true").lower() == "true")
    MULTI_AGENT_TRIGGER_THRESHOLD: int = field(default_factory=lambda: _env_int("MULTI_AGENT_TRIGGER_THRESHOLD", 4))
    MAX_AGENT_COUNT: int = field(default_factory=lambda: _env_int("MAX_AGENT_COUNT", 10))
    AGENT_TASK_TIMEOUT: int = field(default_factory=lambda: _env_int("AGENT_TASK_TIMEOUT", 300))
    # Concurrent sub-agent ceiling. Was hard-coded to 3; lifted now that
    # AGENT_TASK_TIMEOUT is 300s (RTX 3090 + 9B Q8 can sustain 5+ in parallel).
    MAX_PARALLEL_AGENTS: int = field(default_factory=lambda: _env_int("MAX_PARALLEL_AGENTS", 6))
    # Recursive sub-agents: depth 2 = top-level can spawn level-1 sub-agents who can spawn level-2.
    # Threshold gate prevents trivial sub-tasks from cascading; only complex sub-tasks recurse.
    MAX_STRUCTURAL_DEPTH: int = field(default_factory=lambda: _env_int("MAX_STRUCTURAL_DEPTH", 2))
    # Tree-of-thought: when enabled, AgentLoop samples multiple action chains for hard steps
    # and picks the most consistent one. Adds latency proportional to sample count.
    ENABLE_TREE_OF_THOUGHT: bool = field(default_factory=lambda: _env("ENABLE_TREE_OF_THOUGHT", "true").lower() == "true")
    TOT_SAMPLE_N: int = field(default_factory=lambda: _env_int("TOT_SAMPLE_N", 3))
    # Best-of-N for chat path: when a hard reasoning query ends with quality < threshold
    # after the full critique chain, sample N alternative responses at different temperatures
    # and pick the highest-quality one. Bounded — only fires on hard + low-quality.
    ENABLE_BEST_OF_N: bool = field(default_factory=lambda: _env("ENABLE_BEST_OF_N", "true").lower() == "true")
    BEST_OF_N_SAMPLES: int = field(default_factory=lambda: _env_int("BEST_OF_N_SAMPLES", 2))
    # Median answer quality clusters around 0.75 in production. The old 0.70
    # threshold combined with the hard-reasoning-query gate left BEST_OF_N
    # firing on <2% of queries (audit 2026-05-04 #1). 0.65 catches more
    # salvageable mid-quality answers without spamming on already-good ones.
    BEST_OF_N_QUALITY_THRESHOLD: float = field(default_factory=lambda: _env_float("BEST_OF_N_QUALITY_THRESHOLD", 0.65))
    # Hard floor for retrieval injection — chunks scoring below this are
    # dropped entirely (regardless of RETRIEVAL_RELEVANCE_THRESHOLD which
    # governs how many results to return). Above this, chunks reach the
    # prompt without a quality label so the model can't echo "low relevance".
    RETRIEVAL_HARD_FLOOR: float = field(default_factory=lambda: _env_float("RETRIEVAL_HARD_FLOOR", 0.30))

    # Background tasks
    MAX_BACKGROUND_TASKS: int = field(default_factory=lambda: _env_int("MAX_BACKGROUND_TASKS", 5))
    BACKGROUND_TASK_TIMEOUT: int = field(default_factory=lambda: _env_int("BACKGROUND_TASK_TIMEOUT", 300))

    # Auto skill creation
    ENABLE_AUTO_SKILL_CREATION: bool = field(default_factory=lambda: _env("ENABLE_AUTO_SKILL_CREATION", "true").lower() == "true")

    # Autonomous tool creation (self-extending pipeline)
    ENABLE_AUTONOMOUS_TOOL_CREATION: bool = field(default_factory=lambda: _env("ENABLE_AUTONOMOUS_TOOL_CREATION", "true").lower() == "true")
    AUTO_TOOL_CREATION_THRESHOLD: int = field(default_factory=lambda: _env_int("AUTO_TOOL_CREATION_THRESHOLD", 3))

    # Skill import/export signing
    REQUIRE_SIGNED_SKILLS: bool = field(default_factory=lambda: _env("REQUIRE_SIGNED_SKILLS", "true").lower() == "true")

    # Curiosity / autonomy
    ENABLE_CURIOSITY: bool = field(default_factory=lambda: _env("ENABLE_CURIOSITY", "true").lower() == "true")

    # Voice (local Whisper speech-to-text)
    ENABLE_VOICE: bool = field(default_factory=lambda: _env("ENABLE_VOICE", "false").lower() == "true")
    WHISPER_MODEL_SIZE: str = field(default_factory=lambda: _env("WHISPER_MODEL_SIZE", "base"))
    VOICE_MAX_DURATION: int = field(default_factory=lambda: _env_int("VOICE_MAX_DURATION", 300))
    # Text-to-speech (Piper, sovereign/local)
    ENABLE_TTS: bool = field(default_factory=lambda: _env("ENABLE_TTS", "false").lower() == "true")
    TTS_MODEL_PATH: str = field(default_factory=lambda: _env("TTS_MODEL_PATH", "/data/tts/en_US-amy-medium.onnx"))

    # Limits
    # Qwen3.5 supports 128K natively but Ollama's per-VRAM default clamps the
    # 9B Q8 + 24GB-VRAM combo to num_ctx=32768. The earlier 64000 here oversold
    # what the runtime delivers — Ollama silently truncated the prompt. Held
    # at 18000 leaves ~14K for tool results + history + query + response within
    # the 32K window. Bump only if you also raise Ollama's actual num_ctx.
    MAX_SYSTEM_TOKENS: int = field(default_factory=lambda: _env_int("MAX_SYSTEM_TOKENS", 18000))
    MAX_USER_FACTS: int = field(default_factory=lambda: _env_int("MAX_USER_FACTS", 30))
    MAX_KG_FACTS: int = field(default_factory=lambda: _env_int("MAX_KG_FACTS", 5000))
    MAX_LESSON_CANDIDATES: int = field(default_factory=lambda: _env_int("MAX_LESSON_CANDIDATES", 5000))
    MAX_CURIOSITY_PENDING: int = field(default_factory=lambda: _env_int("MAX_CURIOSITY_PENDING", 50))
    MAX_CURIOSITY_ATTEMPTS: int = field(default_factory=lambda: _env_int("MAX_CURIOSITY_ATTEMPTS", 3))
    MAX_CURIOSITY_QUEUE_SIZE: int = field(default_factory=lambda: _env_int("MAX_CURIOSITY_QUEUE_SIZE", 100))
    MAX_CUSTOM_TOOL_CODE_LENGTH: int = field(default_factory=lambda: _env_int("MAX_CUSTOM_TOOL_CODE_LENGTH", 5000))
    MAX_CUSTOM_TOOLS: int = field(default_factory=lambda: _env_int("MAX_CUSTOM_TOOLS", 50))
    RATE_LIMIT_RPM: int = field(default_factory=lambda: _env_int("RATE_LIMIT_RPM", 60))

    # Prompt context limits (how many items of each type in system prompt)
    MAX_KG_FACTS_IN_PROMPT: int = field(default_factory=lambda: _env_int("MAX_KG_FACTS_IN_PROMPT", 20))
    MAX_REFLEXIONS_IN_PROMPT: int = field(default_factory=lambda: _env_int("MAX_REFLEXIONS_IN_PROMPT", 3))
    MAX_SUCCESS_PATTERNS_IN_PROMPT: int = field(default_factory=lambda: _env_int("MAX_SUCCESS_PATTERNS_IN_PROMPT", 2))
    MAX_REFLEXIONS: int = field(default_factory=lambda: _env_int("MAX_REFLEXIONS", 200))

    # Security
    ENABLE_INJECTION_DETECTION: bool = field(default_factory=lambda: _env("ENABLE_INJECTION_DETECTION", "true").lower() == "true")
    # Default false so a fresh localhost install (ports bound to 127.0.0.1 in
    # compose) works key-less out of the box. With an empty NOVA_API_KEY and
    # REQUIRE_AUTH=true, every request fail-closes to 503 — which silently broke
    # the out-of-box experience (cp .env.example .env -> up -> 503 on all chat).
    # Setting NOVA_API_KEY enforces auth regardless; set REQUIRE_AUTH=true to
    # also fail closed when no key is set (do this before any network exposure).
    REQUIRE_AUTH: bool = field(default_factory=lambda: _env("REQUIRE_AUTH", "false").lower() == "true")
    TRUSTED_PROXY: str = field(default_factory=lambda: _env("TRUSTED_PROXY", ""))
    AUTH_MAX_FAILURES: int = field(default_factory=lambda: _env_int("AUTH_MAX_FAILURES", 10))
    AUTH_LOCKOUT_SECONDS: int = field(default_factory=lambda: _env_int("AUTH_LOCKOUT_SECONDS", 300))

    # Query limits
    MAX_QUERY_LENGTH: int = field(default_factory=lambda: _env_int("MAX_QUERY_LENGTH", 50000))

    # --- Tuning parameters ---
    RESPONSE_TOKEN_BUDGET: int = field(default_factory=lambda: _env_int("RESPONSE_TOKEN_BUDGET", 2000))
    RETRIEVAL_RELEVANCE_THRESHOLD: float = field(default_factory=lambda: _env_float("RETRIEVAL_RELEVANCE_THRESHOLD", 0.15))
    TEMPERATURE_DEFAULT: float = field(default_factory=lambda: _env_float("TEMPERATURE_DEFAULT", 0.7))
    # Min blended RRF score for a lesson/fact to survive retrieval. Lowered
    # 0.015 → 0.005 (2026-05-30): a single KEYWORD-only match scores
    # ~0.0139 after the Q-value blend (0.85 × 1/61), so 0.015 silently dropped
    # every keyword-only hit — making lesson retrieval depend entirely on the
    # vector index. When that index is empty/degraded (as found in the WS2
    # audit), the memory loop returned [] for everything. 0.005 keeps real
    # keyword matches (filter already requires ≥2-word overlap) while still
    # rejecting true non-matches (score 0).
    MIN_RRF_SCORE: float = field(default_factory=lambda: _env_float("MIN_RRF_SCORE", 0.005))
    # Max cosine distance for a lesson to pass the semantic (vector) gate in
    # get_relevant_lessons. Cosine: 0=identical, 2=opposite. Raised from the old
    # hardcoded 0.7 to 0.9 so paraphrased queries (low keyword overlap) still
    # surface the relevant lesson — the WS2A "semantic-first" change. The RRF
    # fusion, MIN_RRF_SCORE floor, and 0.40 confidence floor remain as backstops.
    LESSON_VECTOR_MAX_DISTANCE: float = field(default_factory=lambda: _env_float("LESSON_VECTOR_MAX_DISTANCE", 0.9))
    KG_VECTOR_MAX_DISTANCE: float = field(default_factory=lambda: _env_float("KG_VECTOR_MAX_DISTANCE", 0.8))
    DEDUP_JACCARD_THRESHOLD: float = field(default_factory=lambda: _env_float("DEDUP_JACCARD_THRESHOLD", 0.85))
    REFLEXION_DECAY_DAYS: int = field(default_factory=lambda: _env_int("REFLEXION_DECAY_DAYS", 90))
    REFLEXION_DECAY_AMOUNT: float = field(default_factory=lambda: _env_float("REFLEXION_DECAY_AMOUNT", 0.05))
    REFLEXION_DISTANCE_THRESHOLD: float = field(default_factory=lambda: _env_float("REFLEXION_DISTANCE_THRESHOLD", 0.7))
    ENABLE_SEMANTIC_SKILL_MATCHING: bool = field(default_factory=lambda: _env("ENABLE_SEMANTIC_SKILL_MATCHING", "true").lower() == "true")
    SKILL_EMA_ALPHA: float = field(default_factory=lambda: _env_float("SKILL_EMA_ALPHA", 0.15))
    # Cosine similarity threshold for semantic skill matching (higher = stricter)
    SKILL_SEMANTIC_THRESHOLD: float = field(default_factory=lambda: _env_float("SKILL_SEMANTIC_THRESHOLD", 0.55))
    # Days without use before a skill is considered stale for decay/pruning
    SKILL_STALE_DAYS: int = field(default_factory=lambda: _env_int("SKILL_STALE_DAYS", 30))
    INJECTION_SUSPICIOUS_THRESHOLD: float = field(default_factory=lambda: _env_float("INJECTION_SUSPICIOUS_THRESHOLD", 0.3))
    REFLEXION_FAILURE_THRESHOLD: float = field(default_factory=lambda: _env_float("REFLEXION_FAILURE_THRESHOLD", 0.6))
    REFLEXION_SUCCESS_THRESHOLD: float = field(default_factory=lambda: _env_float("REFLEXION_SUCCESS_THRESHOLD", 0.8))
    KG_GRAPH_MAX_FRONTIER: int = field(default_factory=lambda: _env_int("KG_GRAPH_MAX_FRONTIER", 1000))
    AUTH_MAX_TRACKED_IPS: int = field(default_factory=lambda: _env_int("AUTH_MAX_TRACKED_IPS", 10000))

    # Prompt self-modification (opt-in; default off for safety)
    ENABLE_PROMPT_SELF_MOD: bool = field(default_factory=lambda: _env("ENABLE_PROMPT_SELF_MOD", "false").lower() == "true")
    # Max candidate proposals per day per module
    PROMPT_MOD_MAX_PROPOSALS_PER_DAY: int = field(default_factory=lambda: _env_int("PROMPT_MOD_MAX_PROPOSALS_PER_DAY", 2))
    # Max pending candidates per module before blocking new proposals
    PROMPT_MOD_MAX_PENDING: int = field(default_factory=lambda: _env_int("PROMPT_MOD_MAX_PENDING", 3))
    # Max promotions system-wide per day
    PROMPT_MOD_MAX_PROMOTIONS_PER_DAY: int = field(default_factory=lambda: _env_int("PROMPT_MOD_MAX_PROMOTIONS_PER_DAY", 2))
    # Max word-overlap drift from baseline (0.0-1.0; 0.25 ≈ meaningful rephrasing limit)
    PROMPT_MOD_MAX_DRIFT: float = field(default_factory=lambda: _env_float("PROMPT_MOD_MAX_DRIFT", 0.25))
    # Minimum improvement (percentage points) required on target metric to promote
    PROMPT_MOD_MIN_IMPROVEMENT_PP: float = field(default_factory=lambda: _env_float("PROMPT_MOD_MIN_IMPROVEMENT_PP", 2.0))
    # Max regression allowed (pp) in any non-target category before blocking
    PROMPT_MOD_REGRESSION_TOLERANCE_PP: float = field(default_factory=lambda: _env_float("PROMPT_MOD_REGRESSION_TOLERANCE_PP", 1.0))
    # Number of consecutive shadow runs required (candidate must pass K out of this many)
    PROMPT_MOD_STABILITY_RUNS: int = field(default_factory=lambda: _env_int("PROMPT_MOD_STABILITY_RUNS", 3))
    # Max latency overhead before blocking (1.15 = 15% overhead allowed)
    PROMPT_MOD_LATENCY_OVERHEAD_MAX: float = field(default_factory=lambda: _env_float("PROMPT_MOD_LATENCY_OVERHEAD_MAX", 1.15))

    # System access tiers (sandboxed | standard | full | none)
    SYSTEM_ACCESS_LEVEL: str = field(default_factory=lambda: _env("SYSTEM_ACCESS_LEVEL", "sandboxed"))

    # Integrations
    ENABLE_INTEGRATIONS: bool = field(default_factory=lambda: _env("ENABLE_INTEGRATIONS", "true").lower() == "true")

    # Action: Email
    ENABLE_EMAIL_SEND: bool = field(default_factory=lambda: _env("ENABLE_EMAIL_SEND", "false").lower() == "true")
    EMAIL_SMTP_HOST: str = field(default_factory=lambda: _env("EMAIL_SMTP_HOST"))
    EMAIL_SMTP_PORT: int = field(default_factory=lambda: _env_int("EMAIL_SMTP_PORT", 587))
    EMAIL_SMTP_USER: str = field(default_factory=lambda: _env("EMAIL_SMTP_USER"))
    EMAIL_SMTP_PASS: str = field(default_factory=lambda: _env("EMAIL_SMTP_PASS"))
    EMAIL_FROM: str = field(default_factory=lambda: _env("EMAIL_FROM"))
    EMAIL_SMTP_TLS: bool = field(default_factory=lambda: _env("EMAIL_SMTP_TLS", "true").lower() == "true")
    EMAIL_ALLOWED_RECIPIENTS: str = field(default_factory=lambda: _env("EMAIL_ALLOWED_RECIPIENTS"))
    EMAIL_RATE_LIMIT: int = field(default_factory=lambda: _env_int("EMAIL_RATE_LIMIT", 20))

    # Action: Calendar
    ENABLE_CALENDAR: bool = field(default_factory=lambda: _env("ENABLE_CALENDAR", "true").lower() == "true")
    CALENDAR_PATH: str = field(default_factory=lambda: _env("CALENDAR_PATH", "/data/calendar.ics"))

    # Action: Webhooks
    ENABLE_WEBHOOKS: bool = field(default_factory=lambda: _env("ENABLE_WEBHOOKS", "false").lower() == "true")
    WEBHOOK_ALLOWED_URLS: str = field(default_factory=lambda: _env("WEBHOOK_ALLOWED_URLS"))

    # Channels
    DISCORD_TOKEN: str = field(default_factory=lambda: _env("DISCORD_TOKEN"))
    DISCORD_CHANNEL_ID: str = field(default_factory=lambda: _env("DISCORD_CHANNEL_ID"))
    DISCORD_ALLOWED_USERS: str = field(default_factory=lambda: _env("DISCORD_ALLOWED_USERS"))
    TELEGRAM_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_TOKEN"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    TELEGRAM_ALLOWED_USERS: str = field(default_factory=lambda: _env("TELEGRAM_ALLOWED_USERS"))

    # Channel: WhatsApp (Business API or bridge)
    WHATSAPP_API_URL: str = field(default_factory=lambda: _env("WHATSAPP_API_URL"))
    WHATSAPP_API_TOKEN: str = field(default_factory=lambda: _env("WHATSAPP_API_TOKEN"))
    WHATSAPP_VERIFY_TOKEN: str = field(default_factory=lambda: _env("WHATSAPP_VERIFY_TOKEN"))
    WHATSAPP_PHONE_ID: str = field(default_factory=lambda: _env("WHATSAPP_PHONE_ID"))
    WHATSAPP_CHAT_ID: str = field(default_factory=lambda: _env("WHATSAPP_CHAT_ID"))
    WHATSAPP_ALLOWED_USERS: str = field(default_factory=lambda: _env("WHATSAPP_ALLOWED_USERS"))
    WHATSAPP_APP_SECRET: str = field(default_factory=lambda: _env("WHATSAPP_APP_SECRET"))

    # Channel: Signal (via signal-cli REST API)
    SIGNAL_API_URL: str = field(default_factory=lambda: _env("SIGNAL_API_URL"))
    SIGNAL_PHONE_NUMBER: str = field(default_factory=lambda: _env("SIGNAL_PHONE_NUMBER"))
    SIGNAL_CHAT_ID: str = field(default_factory=lambda: _env("SIGNAL_CHAT_ID"))
    SIGNAL_ALLOWED_USERS: str = field(default_factory=lambda: _env("SIGNAL_ALLOWED_USERS"))
    SIGNAL_POLL_INTERVAL: int = field(default_factory=lambda: _env_int("SIGNAL_POLL_INTERVAL", 2))

    # Auth
    API_KEY: str = field(default_factory=lambda: _env("NOVA_API_KEY"))
    ALLOWED_ORIGINS: str = field(default_factory=lambda: _env("ALLOWED_ORIGINS", "http://localhost:5173"))

    # Server
    HOST: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: _env_int("PORT", 8000))

    # Database
    DB_PATH: str = field(default_factory=lambda: _env("DB_PATH", "/data/nova.db"))
    CHROMADB_PATH: str = field(default_factory=lambda: _env("CHROMADB_PATH", "/data/chromadb"))

    # Sensitive field names — redacted in __repr__/__str__ to prevent secret leakage
    _SENSITIVE_FIELDS = frozenset({
        "EMAIL_SMTP_PASS", "DISCORD_TOKEN", "TELEGRAM_TOKEN",
        "WHATSAPP_API_TOKEN", "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET",
        "API_KEY",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value) -> None:
        """Warn on direct attribute mutation after init. Use config.update() instead."""
        if getattr(self, "_initialized", False) and not name.startswith("_"):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Direct config mutation: %s. Use config.update() for runtime changes.", name
            )
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f in self._SENSITIVE_FIELDS and val:
                fields.append(f"{f}='***'")
            else:
                fields.append(f"{f}={val!r}")
        return f"Config({', '.join(fields)})"

    def __str__(self) -> str:
        return self.__repr__()

    def update(self, **kwargs) -> list[str]:
        """Update config values at runtime. Returns validation warnings."""
        for key, value in kwargs.items():
            if not hasattr(self, key) or key.startswith('_'):
                continue
            if key not in _MUTABLE_FIELDS:
                continue
            # Type coerce based on current field type
            current = getattr(self, key)
            if isinstance(current, bool):
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            # Bounds validation for security-sensitive thresholds
            if key == "INJECTION_SUSPICIOUS_THRESHOLD":
                value = max(0.1, min(0.9, float(value)))
            object.__setattr__(self, key, value)
        return self.validate()

    def to_dict(self, redact_sensitive: bool = True) -> dict:
        """Export all config values as dict. Redacts sensitive fields by default."""
        result = {}
        for f in self.__dataclass_fields__:
            if f.startswith('_'):
                continue
            val = getattr(self, f)
            if redact_sensitive and f in self._SENSITIVE_FIELDS and val:
                result[f] = "***"
            else:
                result[f] = val
        return result

    def _save_overrides(self, keys: list[str]) -> None:
        """Save changed keys to overrides file."""
        overrides = {}
        if _OVERRIDES_PATH.exists():
            try:
                overrides = json.loads(_OVERRIDES_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        for key in keys:
            if hasattr(self, key) and not key.startswith('_'):
                overrides[key] = getattr(self, key)
        try:
            _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
        except OSError:
            pass

    def _load_overrides(self) -> None:
        """Apply saved overrides from file.

        Resolution order (most specific wins):
          1. Module-level _OVERRIDES_PATH (mutated by test helpers like
             `monkeypatch.setattr(app.config, '_OVERRIDES_PATH', path)`)
          2. CONFIG_OVERRIDES_PATH env var (re-read each call)
          3. Default `/data/config_overrides.json` (module-level constant)
        """
        # If the module-level path differs from the default, respect it (tests mutate it)
        default_path = Path(os.getenv("CONFIG_OVERRIDES_PATH", "/data/config_overrides.json"))
        path = _OVERRIDES_PATH if _OVERRIDES_PATH != Path("/data/config_overrides.json") else default_path
        if not path.exists():
            return
        try:
            overrides = json.loads(path.read_text())
            # Only load overrides for mutable fields (security: prevent persisted security bypasses)
            filtered = {k: v for k, v in overrides.items() if k in _MUTABLE_FIELDS}
            self.update(**filtered)
        except (json.JSONDecodeError, OSError):
            pass

    def validate(self) -> list[str]:
        """Validate config values. Returns list of warning messages (empty = valid)."""
        warnings = []

        if self.LLM_PROVIDER != "ollama":
            warnings.append(
                f"LLM_PROVIDER must be 'ollama' (cloud providers removed), got: '{self.LLM_PROVIDER}'"
            )

        if not self.OLLAMA_URL.startswith(("http://", "https://")):
            warnings.append(f"OLLAMA_URL must start with http:// or https://, got: {self.OLLAMA_URL}")

        if not (1 <= self.PORT <= 65535):
            warnings.append(f"PORT must be 1-65535, got: {self.PORT}")

        if self.SEARXNG_URL and not self.SEARXNG_URL.startswith(("http://", "https://")):
            warnings.append(f"SEARXNG_URL must start with http:// or https://, got: {self.SEARXNG_URL}")

        if self.MAX_CONTEXT_TOKENS < 1000:
            warnings.append(f"MAX_CONTEXT_TOKENS too low: {self.MAX_CONTEXT_TOKENS} (minimum 1000)")

        if self.SYSTEM_ACCESS_LEVEL.lower() not in ("sandboxed", "standard", "full", "none"):
            warnings.append(
                f"SYSTEM_ACCESS_LEVEL must be sandboxed/standard/full/none, got '{self.SYSTEM_ACCESS_LEVEL}'"
            )

        if not (0 <= self.DIGEST_HOUR <= 23):
            warnings.append(f"DIGEST_HOUR must be 0-23, got: {self.DIGEST_HOUR}")

        if self.USER_TIMEZONE and self.USER_TIMEZONE != "UTC":
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(self.USER_TIMEZONE)
            except (KeyError, ImportError):
                warnings.append(f"USER_TIMEZONE '{self.USER_TIMEZONE}' is not a valid IANA timezone")

        if self.HEARTBEAT_INTERVAL < 1:
            warnings.append(f"HEARTBEAT_INTERVAL must be >= 1, got: {self.HEARTBEAT_INTERVAL}")

        return warnings


# ---------------------------------------------------------------------------
# Singleton management — lazy init with proxy for test swapability
# ---------------------------------------------------------------------------

_config_instance: Config | None = None


def get_config() -> Config:
    """Get the Config singleton. Creates on first access."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance._load_overrides()
    return _config_instance


def reset_config() -> Config:
    """Recreate Config from current env vars. For testing."""
    global _config_instance
    _config_instance = Config()
    _config_instance._load_overrides()
    return _config_instance


class _ConfigProxy:
    """Proxy that delegates to the real Config singleton.

    All modules that do `from app.config import config` get this proxy.
    When tests call reset_config(), every module automatically sees
    the new Config values — no importlib.reload or module-walking needed.
    """

    def __getattr__(self, name: str):
        return getattr(get_config(), name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(get_config(), name, value)

    def __repr__(self) -> str:
        return repr(get_config())

    def __str__(self) -> str:
        return str(get_config())


# Module-level config — a proxy that delegates to the real singleton.
# All existing `from app.config import config` imports work unchanged.
config = _ConfigProxy()
