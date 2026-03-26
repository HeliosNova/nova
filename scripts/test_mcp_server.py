"""Manual integration test for the Nova MCP server.

Starts the MCP server as a subprocess via stdio transport, connects
as an MCP client, lists tools, calls each tool with sample data,
and prints results.

Requires:
  - A Nova database (created automatically if DB_PATH does not exist)
  - The `mcp` package installed

Usage:
  python scripts/test_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _banner(msg: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


async def main() -> int:
    """Run the MCP server integration test."""

    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters
    except ImportError:
        print("ERROR: The 'mcp' package is not installed.")
        print("Install it with: pip install mcp")
        return 1

    runner_path = str(Path(__file__).resolve().parent / "mcp_server_runner.py")
    if not Path(runner_path).exists():
        print(f"ERROR: MCP server runner not found at {runner_path}")
        return 1

    _banner("Nova MCP Server Integration Test")

    # Use a temporary database if DB_PATH is not set
    if not os.environ.get("DB_PATH"):
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="nova_mcp_test_")
        os.environ["DB_PATH"] = str(Path(tmp_dir) / "test.db")
        os.environ.setdefault("CHROMADB_PATH", str(Path(tmp_dir) / "chromadb"))
        print(f"Using temporary DB: {os.environ['DB_PATH']}")

    # Seed some test data into the database
    print("\nSeeding test data...")
    _seed_test_data()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[runner_path],
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )

    print(f"\nStarting MCP server: {sys.executable} {runner_path}")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Session initialized successfully.")

                # --- List tools ---
                _banner("Listing Tools")
                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                print(f"Found {len(tool_names)} tools: {tool_names}")

                expected_tools = {
                    "nova_memory_query",
                    "nova_knowledge_graph",
                    "nova_lessons",
                    "nova_document_search",
                    "nova_facts_about",
                }
                missing = expected_tools - set(tool_names)
                if missing:
                    print(f"FAIL: Missing expected tools: {missing}")
                    return 1
                print("PASS: All 5 expected tools present.")

                # --- Call each tool ---
                test_calls = [
                    ("nova_facts_about", {}, "facts"),
                    ("nova_facts_about", {"category": "fact"}, "filtered facts"),
                    ("nova_knowledge_graph", {"entity": "python"}, "KG query"),
                    ("nova_knowledge_graph", {"entity": ""}, "KG empty entity (expect error)"),
                    ("nova_lessons", {"query": "programming"}, "lessons query"),
                    ("nova_memory_query", {"query": "user"}, "memory query"),
                    ("nova_document_search", {"query": "test"}, "doc search (expect unavailable)"),
                ]

                passed = 0
                failed = 0

                for tool_name, args, description in test_calls:
                    _banner(f"Call: {tool_name} ({description})")
                    print(f"  args: {json.dumps(args)}")

                    try:
                        result = await session.call_tool(tool_name, args)
                        text = result.content[0].text
                        data = json.loads(text)
                        print(f"  Response: {json.dumps(data, indent=2)[:500]}")

                        # Validate structure
                        if isinstance(data, dict):
                            print(f"  PASS: Got valid JSON response")
                            passed += 1
                        else:
                            print(f"  FAIL: Unexpected response type: {type(data)}")
                            failed += 1

                    except Exception as e:
                        print(f"  FAIL: {type(e).__name__}: {e}")
                        failed += 1

                # --- Summary ---
                _banner("Test Summary")
                total = passed + failed
                print(f"  Passed: {passed}/{total}")
                print(f"  Failed: {failed}/{total}")

                return 0 if failed == 0 else 1

    except Exception as e:
        print(f"\nFATAL: Failed to connect to MCP server: {type(e).__name__}: {e}")
        return 1


def _seed_test_data() -> None:
    """Insert minimal test data into the database for the integration test."""
    try:
        from app.database import SafeDB

        db_path = os.environ.get("DB_PATH", "/data/nova.db")
        db = SafeDB(db_path)
        db.init_schema()

        # User facts
        db.execute(
            "INSERT OR IGNORE INTO user_facts (key, value, source, confidence, category) "
            "VALUES (?, ?, ?, ?, ?)",
            ("name", "Test User", "test", 1.0, "fact"),
        )
        db.execute(
            "INSERT OR IGNORE INTO user_facts (key, value, source, confidence, category) "
            "VALUES (?, ?, ?, ?, ?)",
            ("preferred_editor", "VSCode", "test", 0.9, "preference"),
        )

        # KG facts
        db.execute(
            "INSERT OR IGNORE INTO kg_facts (subject, predicate, object, confidence, source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("python", "is_a", "programming language", 0.9, "test"),
        )
        db.execute(
            "INSERT OR IGNORE INTO kg_facts (subject, predicate, object, confidence, source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("python", "created_by", "guido van rossum", 0.95, "test"),
        )

        # A lesson
        db.execute(
            "INSERT OR IGNORE INTO lessons "
            "(topic, wrong_answer, correct_answer, lesson_text, context, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "Python creator",
                "Python was created by James Gosling",
                "Python was created by Guido van Rossum",
                "Python was created by Guido van Rossum, not James Gosling",
                "programming languages",
                0.8,
            ),
        )

        # A conversation with messages
        import uuid
        conv_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, "Test Conversation"),
        )
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), conv_id, "user", "Tell me about Python programming"),
        )
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), conv_id, "assistant", "Python is a versatile programming language created by Guido van Rossum."),
        )

        print(f"  Seeded: 2 user facts, 2 KG facts, 1 lesson, 1 conversation")
        db.close()

    except Exception as e:
        print(f"  Warning: Failed to seed test data: {e}")
        print(f"  (Tests will run against whatever data exists in the DB)")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    sys.exit(exit_code)
