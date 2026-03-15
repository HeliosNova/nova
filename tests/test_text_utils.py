"""Tests for text_utils — stemming, normalization."""

from app.core.text_utils import simple_stem, normalize_words, content_words


class TestSimpleStem:
    def test_irregular_demonyms(self):
        assert simple_stem("french") == "france"
        assert simple_stem("british") == "britain"
        assert simple_stem("japanese") == "japan"
        assert simple_stem("american") == "america"

    def test_irregular_misc(self):
        assert simple_stem("children") == "child"
        assert simple_stem("better") == "good"
        assert simple_stem("analyses") == "analysis"

    def test_suffix_stripping(self):
        assert simple_stem("optimization") == "optimize"
        assert simple_stem("running") == "run"
        assert simple_stem("quickly") == "quick"
        assert simple_stem("databases") == "database"
        assert simple_stem("countries") == "country"

    def test_short_words_unchanged(self):
        assert simple_stem("go") == "go"
        assert simple_stem("the") == "the"
        assert simple_stem("sql") == "sql"

    def test_case_insensitive(self):
        assert simple_stem("French") == "france"
        assert simple_stem("RUNNING") == "run"

    def test_no_over_strip(self):
        # Stem should not strip below 3 chars
        stem = simple_stem("axes")
        assert len(stem) >= 3


class TestNormalizeWords:
    def test_basic(self):
        words = normalize_words("Hello world, this is a test!")
        assert "hello" in words  # no suffix to strip
        assert "test" in words
        assert "world" in words
        # Stop words removed
        assert "this" not in words
        assert "is" not in words
        assert "a" not in words

    def test_stemming_enabled(self):
        words = normalize_words("French optimization")
        assert "france" in words
        assert "optimize" in words

    def test_stemming_disabled(self):
        words = normalize_words("French optimization", stem=False)
        assert "french" in words
        assert "optimization" in words
        assert "france" not in words

    def test_min_length(self):
        words = normalize_words("I am a big dog", min_length=2)
        assert "big" in words or "dog" in words
        # Single-char words should be filtered by min_length + stop words

    def test_cross_match(self):
        """'France' and 'French' should produce the same stem."""
        words_a = normalize_words("France is lovely")
        words_b = normalize_words("French cuisine")
        assert words_a & words_b  # Non-empty intersection


class TestContentWords:
    def test_delegates_to_normalize(self):
        result = content_words("Hello world")
        assert isinstance(result, set)
        assert "hello" in result or any(w for w in result)
