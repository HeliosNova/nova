"""Background task manager — runs async tasks without blocking the main conversation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Coroutine

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    id: str
    description: str
    status: str  # pending, running, complete, failed, cancelled
    result: str | None = None
    error: str | None = None
    partial_result: str | None = None
    created_at: str = ""
    completed_at: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class TaskManager:
    """Manages background async tasks with concurrency limits."""

    def __init__(self, max_concurrent: int = 5, task_timeout: int = 300):
        self.max_concurrent = max_concurrent
        self.task_timeout = task_timeout
        self._tasks: dict[str, BackgroundTask] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def submit(
        self,
        coro: Coroutine,
        description: str,
        partial_collector: list[str] | None = None,
    ) -> str:
        """Submit a coroutine as a background task. Returns task ID.

        If *partial_collector* is provided (a mutable list of strings), its
        contents are joined and saved as ``partial_result`` when the task
        fails, giving callers whatever was collected before the error.
        """
        # Check concurrency limit (non-blocking check for fast rejection)
        active = sum(1 for t in self._tasks.values() if t.status in ("pending", "running"))
        if active >= self.max_concurrent:
            raise RuntimeError(f"Max background tasks ({self.max_concurrent}) reached")

        task_id = str(uuid.uuid4())[:8]
        bg = BackgroundTask(
            id=task_id,
            description=description,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        async def _run():
            async with self._semaphore:
                bg.status = "running"
                try:
                    result = await asyncio.wait_for(coro, timeout=self.task_timeout)
                    bg.result = str(result) if result else "Completed"
                    bg.status = "complete"
                except asyncio.TimeoutError:
                    bg.error = f"Task timed out after {self.task_timeout}s"
                    bg.status = "failed"
                except asyncio.CancelledError:
                    bg.status = "cancelled"
                except Exception as e:
                    bg.error = str(e)
                    bg.status = "failed"
                    logger.error("[TaskManager] Task %s failed: %s", task_id, e)
                finally:
                    # Capture partial results on non-success
                    if partial_collector and bg.status in ("failed", "cancelled"):
                        partial = "".join(partial_collector)
                        if partial.strip():
                            bg.partial_result = partial[:3000]
                    bg.completed_at = datetime.now(timezone.utc).isoformat()
                    # Release coroutine frame to free memory (Phase 5.9)
                    bg._task = None

        bg._task = asyncio.create_task(_run())
        self._tasks[task_id] = bg

        # Prune old completed tasks (keep last 50)
        completed = [t for t in self._tasks.values() if t.status in ("complete", "failed", "cancelled")]
        if len(completed) > 50:
            for old in sorted(completed, key=lambda t: t.created_at)[:len(completed) - 50]:
                del self._tasks[old.id]

        logger.info("[TaskManager] Submitted task %s: %s", task_id, description)
        return task_id

    def get_status(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 20) -> list[BackgroundTask]:
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def cancel(self, task_id: str) -> bool:
        bg = self._tasks.get(task_id)
        if not bg or bg.status not in ("pending", "running"):
            return False
        if bg._task and not bg._task.done():
            bg._task.cancel()
        bg.status = "cancelled"
        bg.completed_at = datetime.now(timezone.utc).isoformat()
        return True

    async def cancel_all(self):
        tasks_to_cancel = []
        for bg in self._tasks.values():
            if bg._task and not bg._task.done():
                bg._task.cancel()
                tasks_to_cancel.append(bg._task)
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
