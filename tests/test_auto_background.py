"""Tests for auto-background promotion and TaskManager.track_existing()."""

import asyncio

import pytest

from app.core.task_manager import BackgroundTask, TaskManager


# ---------------------------------------------------------------------------
# track_existing()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_existing_completes():
    """Task that completes normally is tracked and updated."""
    tm = TaskManager(max_concurrent=5, task_timeout=300)

    async def slow_work():
        await asyncio.sleep(0.1)
        return ("tool output", None)

    task = asyncio.create_task(slow_work())
    task_id = tm.track_existing(task, "test task")
    assert task_id  # Non-empty string

    bg = tm.get_status(task_id)
    assert bg is not None
    assert bg.status == "running"

    # Wait for completion
    await asyncio.sleep(0.3)
    bg = tm.get_status(task_id)
    assert bg.status == "complete"
    assert bg.result is not None


@pytest.mark.asyncio
async def test_track_existing_fails():
    """Task that raises is tracked as failed."""
    tm = TaskManager(max_concurrent=5, task_timeout=300)

    async def failing_work():
        raise ValueError("boom")

    task = asyncio.create_task(failing_work())
    task_id = tm.track_existing(task, "failing task")
    assert task_id

    await asyncio.sleep(0.2)
    bg = tm.get_status(task_id)
    assert bg.status == "failed"
    assert "boom" in bg.error


@pytest.mark.asyncio
async def test_track_existing_at_capacity():
    """Returns empty string when at max capacity."""
    tm = TaskManager(max_concurrent=1, task_timeout=300)

    # Fill capacity
    async def long_work():
        await asyncio.sleep(10)

    task1 = asyncio.create_task(long_work())
    tm.track_existing(task1, "task 1")

    # Second track should fail (at capacity)
    task2 = asyncio.create_task(long_work())
    task_id = tm.track_existing(task2, "task 2")
    assert task_id == ""

    # Cleanup
    task1.cancel()
    task2.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass
    try:
        await task2
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_track_existing_auto_prune():
    """Old completed tasks are pruned when exceeding 50."""
    tm = TaskManager(max_concurrent=100, task_timeout=300)

    # Create 55 tasks, let them complete, then add more to trigger pruning
    for i in range(55):
        async def instant():
            return "done"
        task = asyncio.create_task(instant())
        tm.track_existing(task, f"task {i}")

    # Wait for all to complete so they're counted as completed
    await asyncio.sleep(0.3)

    # Now add 5 more — this should trigger pruning of oldest completed
    for i in range(55, 60):
        async def instant2():
            return "done"
        task = asyncio.create_task(instant2())
        tm.track_existing(task, f"task {i}")

    await asyncio.sleep(0.3)

    all_tasks = tm.list_tasks(limit=100)
    # Some pruning should have occurred (not all 60 retained)
    assert len(all_tasks) <= 56  # 50 kept + up to 5 new + margin
