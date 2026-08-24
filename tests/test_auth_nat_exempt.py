"""Auth NAT-collapse lockout exemption (audit 2026-08-22).

Under Docker NAT every host->published-port request collapses to the bridge
gateway/container IP, so locking that address out on repeated auth failures
DoSes the owner with zero security benefit (it can't tell clients apart — the
live auth_lockouts row was 172.18.0.8). Loopback + 172.16/12 are exempt from the
failure-lockout; genuine LAN (10/8, 192.168/16) and public IPs still lock out.
"""

import importlib

import pytest
from fastapi import HTTPException


def _fresh_auth():
    import app.auth
    importlib.reload(app.auth)
    return app.auth


@pytest.mark.parametrize("ip", ["172.18.0.8", "172.17.0.1", "127.0.0.1", "::1"])
def test_nat_collapsed_ip_never_locks_out(ip):
    auth = _fresh_auth()
    for _ in range(auth.config.AUTH_MAX_FAILURES + 5):
        auth._record_failure(ip)
    auth._check_rate_limit(ip)  # must NOT raise — exempt address
    assert ip not in auth._lockouts


def test_real_public_ip_still_locks_out():
    auth = _fresh_auth()
    ip = "203.0.113.7"  # public documentation range — a real client via a proxy
    auth._lockouts.pop(ip, None)
    auth._auth_failures.pop(ip, None)
    for _ in range(auth.config.AUTH_MAX_FAILURES):
        auth._record_failure(ip)
    with pytest.raises(HTTPException) as exc:
        auth._check_rate_limit(ip)
    assert exc.value.status_code == 429


def test_real_lan_ip_still_locks_out():
    # 10/8 is NAT-collapse-adjacent but a genuine client range behind a proxy;
    # it must remain protected (this also guards the existing 10.0.0.x lockout tests).
    auth = _fresh_auth()
    ip = "10.0.0.99"
    auth._lockouts.pop(ip, None)
    auth._auth_failures.pop(ip, None)
    for _ in range(auth.config.AUTH_MAX_FAILURES):
        auth._record_failure(ip)
    with pytest.raises(HTTPException) as exc:
        auth._check_rate_limit(ip)
    assert exc.value.status_code == 429
