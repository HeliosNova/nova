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
    ("ities", "ity"), ("ying", "y"), ("ling", "l"),
    ("ating", "ate"), ("izing", "ize"),
    ("tion", "te"), ("sion", "se"),
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
