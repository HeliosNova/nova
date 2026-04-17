"""One-off backfill: reconcile SQLite FTS5 index with ChromaDB documents collection.

Background: an earlier ingest bug (or a mix of partial writes) left FTS5 heavily
under-populated — e.g. ~5 chunks for ~110 ChromaDB docs. BM25 is half of hybrid
retrieval; when FTS5 is empty, recall degrades silently.

This script re-inserts every ChromaDB chunk that is missing from chunks_fts.
It's idempotent: chunks already in FTS5 are skipped. Safe to re-run.

Usage:
    python scripts/backfill_fts5.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from app.config import config  # noqa: F401  (ensures config is initialized)
from app.core.retriever import Retriever
from app.database import get_db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only, do not insert.",
    )
    args = parser.parse_args()

    retriever = Retriever(db=get_db())

    if args.dry_run:
        try:
            collection = retriever._get_collection()
            chroma_n = collection.count()
        except Exception as e:
            print(f"ChromaDB unavailable: {e}")
            return 2
        fts_row = retriever._db.fetchone("SELECT count(*) AS c FROM chunks_fts")
        fts_n = int(fts_row["c"]) if fts_row else 0
        print(f"ChromaDB chunks: {chroma_n}")
        print(f"FTS5 chunks:     {fts_n}")
        print(f"Missing from FTS5: {max(0, chroma_n - fts_n)}")
        return 0

    report = retriever.backfill_fts5()
    if "error" in report:
        print(f"Backfill failed: {report['error']}")
        return 2

    print("Backfill report:")
    print(f"  ChromaDB chunks: {report['chromadb_chunks']}")
    print(f"  FTS5 before:     {report['fts5_before']}")
    print(f"  FTS5 after:      {report['fts5_after']}")
    print(f"  Inserted:        {report['inserted']}")
    print(f"  Skipped:         {report['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
