"""Item 67: Test 10 concurrent asyncio tasks writing to the same SQLite table.

Uses the existing SafeDB from app/database.py. Verifies no locking errors occur.
"""

from __future__ import annotations

import asyncio

import pytest

from app.database import SafeDB


class TestConcurrentDB:
    """Test concurrent access to SafeDB."""

    @pytest.mark.asyncio
    async def test_10_concurrent_writes(self, tmp_path):
        """10 concurrent tasks writing to the same table should not produce locking errors."""
        db = SafeDB(str(tmp_path / "concurrent_test.db"))
        db.init_schema()

        async def write_message(i: int):
            """Write a message to the database in a background thread."""
            conv_id = f"conv-{i}"
            # Create conversation
            await asyncio.to_thread(
                db.execute,
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conv_id, f"Test Conversation {i}"),
            )
            # Add message
            await asyncio.to_thread(
                db.execute,
                "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                (f"msg-{i}", conv_id, "user", f"Message from task {i}"),
            )
            return i

        # Run 10 concurrent writes
        results = await asyncio.gather(*[write_message(i) for i in range(10)])

        assert sorted(results) == list(range(10))

        # Verify all writes succeeded
        rows = db.fetchall("SELECT COUNT(*) as c FROM conversations")
        assert rows[0]["c"] == 10

        rows = db.fetchall("SELECT COUNT(*) as c FROM messages")
        assert rows[0]["c"] == 10

        db.close()

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, tmp_path):
        """Mixed concurrent reads and writes should not deadlock or error."""
        db = SafeDB(str(tmp_path / "mixed_test.db"))
        db.init_schema()

        # Pre-populate some data
        for i in range(5):
            db.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (f"pre-{i}", f"Pre-existing {i}"),
            )

        async def read_task(task_id: int):
            rows = await asyncio.to_thread(
                db.fetchall,
                "SELECT COUNT(*) as c FROM conversations",
            )
            return ("read", task_id, rows[0]["c"])

        async def write_task(task_id: int):
            await asyncio.to_thread(
                db.execute,
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (f"write-{task_id}", f"Written {task_id}"),
            )
            return ("write", task_id, 1)

        # Mix 5 reads and 5 writes
        tasks = []
        for i in range(5):
            tasks.append(read_task(i))
            tasks.append(write_task(i))

        results = await asyncio.gather(*tasks)
        assert len(results) == 10

        # All writes should have succeeded
        write_count = sum(1 for r in results if r[0] == "write")
        assert write_count == 5

        # Total conversations should be 5 (pre) + 5 (written) = 10
        rows = db.fetchall("SELECT COUNT(*) as c FROM conversations")
        assert rows[0]["c"] == 10

        db.close()

    @pytest.mark.asyncio
    async def test_concurrent_transactions(self, tmp_path):
        """Concurrent transactions should not cause integrity errors."""
        db = SafeDB(str(tmp_path / "tx_test.db"))
        db.init_schema()

        async def transactional_write(i: int):
            def _do_tx():
                with db.transaction() as tx:
                    tx.execute(
                        "INSERT INTO conversations (id, title) VALUES (?, ?)",
                        (f"tx-{i}", f"TX Conversation {i}"),
                    )
                    tx.execute(
                        "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                        (f"tx-msg-{i}", f"tx-{i}", "user", f"TX message {i}"),
                    )

            await asyncio.to_thread(_do_tx)
            return i

        results = await asyncio.gather(*[transactional_write(i) for i in range(10)])
        assert sorted(results) == list(range(10))

        # Verify all data written correctly
        convs = db.fetchall("SELECT COUNT(*) as c FROM conversations")
        assert convs[0]["c"] == 10

        msgs = db.fetchall("SELECT COUNT(*) as c FROM messages")
        assert msgs[0]["c"] == 10

        db.close()

    @pytest.mark.asyncio
    async def test_concurrent_user_fact_updates(self, tmp_path):
        """Concurrent user fact updates should be serialized without errors."""
        db = SafeDB(str(tmp_path / "facts_test.db"))
        db.init_schema()

        from app.core.memory import UserFactStore
        facts = UserFactStore(db)

        async def write_fact(i: int):
            await asyncio.to_thread(
                facts.set,
                f"fact_{i}",
                f"value_{i}",
                "extracted",
                0.9,
            )
            return i

        results = await asyncio.gather(*[write_fact(i) for i in range(10)])
        assert sorted(results) == list(range(10))

        # Verify all facts written
        all_facts = facts.get_all()
        assert len(all_facts) == 10

        db.close()

    @pytest.mark.asyncio
    async def test_concurrent_lesson_writes(self, tmp_path):
        """Concurrent lesson writes should not produce locking errors."""
        db = SafeDB(str(tmp_path / "lessons_test.db"))
        db.init_schema()

        async def write_lesson(i: int):
            await asyncio.to_thread(
                db.execute,
                "INSERT INTO lessons (topic, correct_answer, confidence) VALUES (?, ?, ?)",
                (f"topic_{i}", f"answer_{i}", 0.8),
            )
            return i

        results = await asyncio.gather(*[write_lesson(i) for i in range(10)])
        assert sorted(results) == list(range(10))

        rows = db.fetchall("SELECT COUNT(*) as c FROM lessons")
        assert rows[0]["c"] == 10

        db.close()
