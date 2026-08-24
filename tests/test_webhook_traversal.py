"""Webhook allowlist path-traversal rejection (audit 2026-08-22).

A '..' segment passes the prefix check (its first char is '/') and can then
escape the allowed path once the receiving server normalizes it
(/webhooks/../admin -> /admin). The allowlist must reject traversal outright.
"""

from unittest.mock import MagicMock, patch

from app.tools.action_webhook import _is_url_allowed


def _check(url: str, allowlist: str) -> bool:
    cfg = MagicMock()
    cfg.WEBHOOK_ALLOWED_URLS = allowlist
    with patch("app.tools.action_webhook.config", cfg):
        return _is_url_allowed(url)


ALLOW = "https://api.example.com/webhooks/"


def test_allows_legit_prefix():
    assert _check("https://api.example.com/webhooks/deploy", ALLOW) is True


def test_rejects_path_traversal():
    assert _check("https://api.example.com/webhooks/../admin", ALLOW) is False


def test_rejects_encoded_parent_segment():
    # A trailing '..' segment must also be rejected.
    assert _check("https://api.example.com/webhooks/sub/..", ALLOW) is False


def test_rejects_other_host():
    assert _check("https://evil.example.com/webhooks/x", ALLOW) is False


def test_empty_allowlist_blocks_all():
    assert _check("https://api.example.com/webhooks/deploy", "") is False
