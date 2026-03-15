"""Tests for background task manager and tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.core.task_manager import BackgroundTask, TaskManager
from app.tools.background_task import BackgroundTaskTool


# ---------------------------------------------------------------------------
# TaskManager unit tests
# ---------------------------------------------------------------------------

class TestTaskManagerSubmit:
    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def simple():
            return "done"

        task_id = tm.submit(simple(), description="test task")
        assert isinstance(task_id, str)
        assert len(task_id) == 8

    @pytest.mark.asyncio
    async def test_get_status(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def simple():
            return "done"

        task_id = tm.submit(simple(), description="test task")
        bg = tm.get_status(task_id)
        assert bg is not None
        assert bg.id == task_id
        assert bg.description == "test task"

    @pytest.mark.asyncio
    async def test_get_status_unknown(self):
        tm = TaskManager()
        assert tm.get_status("nonexistent") is None


class TestTaskManagerCompletion:
    @pytest.mark.asyncio
    async def test_task_completes_successfully(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def compute():
            return "42"

        task_id = tm.submit(compute(), description="compute answer")
        # Let the event loop process the task
        await asyncio.sleep(0.1)

        bg = tm.get_status(task_id)
        assert bg.status == "complete"
        assert bg.result == "42"
        assert bg.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_completes_none_result(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def no_return():
            pass

        task_id = tm.submit(no_return(), description="void task")
        await asyncio.sleep(0.1)

        bg = tm.get_status(task_id)
        assert bg.status == "complete"
        assert bg.result == "Completed"

    @pytest.mark.asyncio
    async def test_task_failure(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def failing():
            raise ValueError("something broke")

        task_id = tm.submit(failing(), description="bad task")
        await asyncio.sleep(0.1)

        bg = tm.get_status(task_id)
        assert bg.status == "failed"
        assert "something broke" in bg.error
        assert bg.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        tm = TaskManager(max_concurrent=5, task_timeout=1)

        async def slow():
            await asyncio.sleep(10)

        task_id = tm.submit(slow(), description="slow task")
        await asyncio.sleep(1.5)

        bg = tm.get_status(task_id)
        assert bg.status == "failed"
        assert "timed out" in bg.error


class TestTaskManagerCancellation:
    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        tm = TaskManager(max_concurrent=5, task_timeout=60)

        async def long_running():
            await asyncio.sleep(60)

        task_id = tm.submit(long_running(), description="long task")
        await asyncio.sleep(0.05)

        assert tm.cancel(task_id) is True
        bg = tm.get_status(task_id)
        assert bg.status == "cancelled"
        assert bg.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def quick():
            return "fast"

        task_id = tm.submit(quick(), description="quick task")
        await asyncio.sleep(0.1)

        assert tm.cancel(task_id) is False

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        tm = TaskManager(max_concurrent=5, task_timeout=60)

        async def long_running():
            await asyncio.sleep(60)

        tm.submit(long_running(), description="task 1")
        tm.submit(long_running(), description="task 2")
        await asyncio.sleep(0.05)

        await tm.cancel_all()
        # Give cancellation time to propagate
        await asyncio.sleep(0.1)


class TestTaskManagerLimits:
    @pytest.mark.asyncio
    async def test_max_concurrent(self):
        tm = TaskManager(max_concurrent=2, task_timeout=60)

        async def long_running():
            await asyncio.sleep(60)

        tm.submit(long_running(), description="task 1")
        tm.submit(long_running(), description="task 2")

        with pytest.raises(RuntimeError, match="Max background tasks"):
            tm.submit(long_running(), description="task 3")

        # Cleanup
        await tm.cancel_all()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def quick():
            return "ok"

        tm.submit(quick(), description="task A")
        tm.submit(quick(), description="task B")
        await asyncio.sleep(0.1)

        tasks = tm.list_tasks()
        assert len(tasks) == 2
        descriptions = {t.description for t in tasks}
        assert "task A" in descriptions
        assert "task B" in descriptions

    @pytest.mark.asyncio
    async def test_list_tasks_limit(self):
        tm = TaskManager(max_concurrent=5, task_timeout=10)

        async def quick():
            return "ok"

        for i in range(5):
            tm.submit(quick(), description=f"task {i}")
        await asyncio.sleep(0.1)

        tasks = tm.list_tasks(limit=3)
        assert len(tasks) == 3


# ---------------------------------------------------------------------------
# BackgroundTaskTool tests
# ---------------------------------------------------------------------------

class TestBackgroundTaskTool:
    @pytest.fixture
    def tool(self):
        return BackgroundTaskTool()

    @pytest.mark.asyncio
    async def test_no_action(self, tool):
        r = await tool.execute()
        assert not r.success
        assert "No action" in r.error

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        tm = TaskManager()
        svc = MagicMock()
        svc.task_manager = tm
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="explode")
        assert not r.success
        assert "Unknown action" in r.error

    @pytest.mark.asyncio
    async def test_list_empty(self, tool):
        tm = TaskManager()
        svc = MagicMock()
        svc.task_manager = tm
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="list")
        assert r.success
        assert "No background tasks" in r.output

    @pytest.mark.asyncio
    async def test_status_no_task_id(self, tool):
        tm = TaskManager()
        svc = MagicMock()
        svc.task_manager = tm
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="status")
        assert not r.success
        assert "No task_id" in r.error

    @pytest.mark.asyncio
    async def test_cancel_no_task_id(self, tool):
        tm = TaskManager()
        svc = MagicMock()
        svc.task_manager = tm
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="cancel")
        assert not r.success
        assert "No task_id" in r.error

    @pytest.mark.asyncio
    async def test_submit_no_task(self, tool):
        tm = TaskManager()
        svc = MagicMock()
        svc.task_manager = tm
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="submit")
        assert not r.success
        assert "No task" in r.error

    @pytest.mark.asyncio
    async def test_services_not_initialized(self, tool):
        with patch("app.core.brain.get_services", side_effect=RuntimeError("not init")):
            r = await tool.execute(action="list")
        assert not r.success
        assert "not initialized" in r.error

    @pytest.mark.asyncio
    async def test_task_manager_none(self, tool):
        svc = MagicMock()
        svc.task_manager = None
        with patch("app.core.brain.get_services", return_value=svc):
            r = await tool.execute(action="list")
        assert not r.success
        assert "not available" in r.error
