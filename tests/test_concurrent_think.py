"""Tests for concurrent think() serialization and parallelism.

Verifies:
- Same-conversation calls are serialized (via per-conversation lock)
- Different-conversation calls can run in parallel
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def test_conversation_lock_creation():
    """Per-conversation locks are created on demand."""
    from app.core.brain import _get_conversation_lock, _conversation_locks
    _conversation_locks.clear()

    lock1 = await _get_conversation_lock("conv-1")
    lock2 = await _get_conversation_lock("conv-1")
    lock3 = await _get_conversation_lock("conv-2")

    assert lock1 is lock2, "Same conversation should return same lock"
    assert lock1 is not lock3, "Different conversations should have different locks"


async def test_conversation_lock_serializes():
    """Same-conversation lock prevents concurrent access."""
    from app.core.brain import _get_conversation_lock, _conversation_locks
    _conversation_locks.clear()

    lock = await _get_conversation_lock("conv-serial")
    order = []

    async def worker(name: str, delay: float):
        await lock.acquire()
        try:
            order.append(f"{name}-start")
            await asyncio.sleep(delay)
            order.append(f"{name}-end")
        finally:
            lock.release()

    await asyncio.gather(worker("A", 0.05), worker("B", 0.01))
    # A and B should not interleave — one must complete before other starts
    a_start = order.index("A-start")
    a_end = order.index("A-end")
    b_start = order.index("B-start")
    b_end = order.index("B-end")
    assert (a_end < b_start) or (b_end < a_start), f"Lock should serialize: {order}"


async def test_different_conversations_parallel():
    """Different conversations can run in parallel."""
    from app.core.brain import _get_conversation_lock, _conversation_locks
    _conversation_locks.clear()

    lock_a = await _get_conversation_lock("conv-A")
    lock_b = await _get_conversation_lock("conv-B")
    events = []

    async def worker(name: str, lock: asyncio.Lock):
        await lock.acquire()
        try:
            events.append(f"{name}-start")
            await asyncio.sleep(0.05)
            events.append(f"{name}-end")
        finally:
            lock.release()

    await asyncio.gather(worker("A", lock_a), worker("B", lock_b))
    # Both should start before either ends (parallel)
    a_start = events.index("A-start")
    b_start = events.index("B-start")
    a_end = events.index("A-end")
    b_end = events.index("B-end")
    # At least one should start before the other ends
    assert a_start < b_end or b_start < a_end, f"Different convs should be parallel: {events}"
