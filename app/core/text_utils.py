"""Shared text utilities — stop words, normalization, stemming.

Single source of truth for text processing used by learning, retriever,
reflexion, and kg modules.
"""

from __future__ import annotations

import re

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "myself", "you", "your", "he", "him",
    "his", "she", "her", "we", "they", "them", "their", "about",
    "yet", "either", "neither", "also", "still", "already", "even",
    "much", "many", "well", "back", "up", "out",
})


# ---------------------------------------------------------------------------
# Simple stemming — suffix stripping + irregular map
# ---------------------------------------------------------------------------

_IRREGULAR_STEMS: dict[str, str] = {
    # Demonyms / nationalities
    "french": "france", "british": "britain", "english": "england",
    "german": "germany", "spanish": "spain", "italian": "italy",
    "japanese": "japan", "chinese": "china", "russian": "russia",
    "american": "america", "canadian": "canada", "australian": "australia",
    "mexican": "mexico", "brazilian": "brazil", "indian": "india",
    "korean": "korea", "swedish": "sweden", "norwegian": "norway",
    "danish": "denmark", "finnish": "finland", "polish": "poland",
    "dutch": "netherlands", "swiss": "switzerland", "turkish": "turkey",
    "egyptian": "egypt", "greek": "greece", "irish": "ireland",
    "scottish": "scotland", "portuguese": "portugal", "thai": "thailand",
    "vietnamese": "vietnam", "argentine": "argentina", "chilean": "chile",
    "colombian": "colombia", "peruvian": "peru",
    # Common irregulars
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot",
    "data": "datum", "criteria": "criterion", "analyses": "analysis",
    "indices": "index", "matrices": "matrix", "vertices": "vertex",
}

# Ordered longest-first so we strip the longest matching suffix
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
    ("fulness", "ful"), ("iveness", "ive"), ("ousness", "ous"),
    ("ement", ""), ("ment", ""), ("ness", ""),
    ("ities", "ity"), ("ying", "y"),
    ("ating", "ate"), ("izing", "ize"),
    ("ation", "ate"), ("ssion", "ss"),
    ("ious", ""), ("eous", ""), ("ous", ""),
    ("ible", ""), ("able", ""),
    ("ive", ""), ("ize", "ize"),  # keep -ize so "optimize" stays "optimize"
    ("ing", ""), ("ied", "y"), ("ies", "y"),
    ("ful", ""), ("ist", ""), ("ism", ""),
    ("ers", ""), ("ity", ""),
    ("ly", ""), ("ed", ""), ("er", ""),
    ("ves", "ve"),  # lives -> live, knives -> knive
    ("es", "e"),    # codes -> code, wines -> wine (keep the trailing e)
    ("al", ""), ("s", ""),
]


def simple_stem(word: str) -> str:
    """Lightweight stem: irregular map first, then suffix stripping.

    Not a full Porter stemmer — just enough to match "France" with "French",
    "optimize" with "optimization", etc.
    """
    w = word.lower()
    if len(w) <= 3:
        return w

    # Irregular map
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]

    # Suffix stripping — only strip if remainder is >= 3 chars
    for suffix, replacement in _SUFFIX_RULES:
        if w.endswith(suffix):
            stem = w[: -len(suffix)] + replacement
            if len(stem) >= 3:
                # De-duplicate trailing consonant (running -> runn -> run)
                if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                    stem = stem[:-1]
                return stem
            break  # Don't try shorter suffixes if this one left too short

    return w


def normalize_words(text: str, *, min_length: int = 1, stem: bool = True) -> set[str]:
    """Lowercase, strip punctuation, return word set minus stop words.

    Args:
        text: Input text to normalize.
        min_length: Minimum word length to include (default 1).
        stem: Apply simple_stem to each word (default True).
    """
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    result = {w for w in words if len(w) >= min_length and w not in STOP_WORDS}
    if stem:
        result = {simple_stem(w) for w in result}
    return result


def content_words(text: str) -> set[str]:
    """Lowercase words, strip punctuation, remove stop words, stem. Shorthand for normalize_words."""
    return normalize_words(text)


def detect_image_mime(base64_data: str) -> str:
    """Detect image MIME type from base64 data URI prefix or magic bytes.

    Returns a MIME type string like 'image/png', 'image/jpeg', etc.
    Falls back to 'image/png' if detection fails.
    """
    # Check data URI prefix first (e.g., "data:image/jpeg;base64,...")
    if base64_data.startswith("data:"):
        try:
            mime = base64_data.split(";")[0].split(":")[1]
            if mime.startswith("image/"):
                return mime
        except (IndexError, ValueError):
            pass
        # Strip the data URI prefix for magic byte detection
        if ",," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        elif "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

    # Detect from decoded magic bytes
    import base64
    try:
        header = base64.b64decode(base64_data[:32])
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if header[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return "image/webp"
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
    except Exception:
        pass

    return "image/png"  # safe default


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for English, ~1.5 chars/token for CJK."""
    if not text:
        return 0
    cjk_count = sum(
        1 for c in text
        if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af'
    )
    non_cjk = len(text) - cjk_count
    return non_cjk // 4 + int(cjk_count / 1.5)
