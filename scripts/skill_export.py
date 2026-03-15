"""CLI for skill import/export with signature verification.

Usage:
    python scripts/skill_export.py export --output skills.json [--sign-key key.hex]
    python scripts/skill_export.py import --input skills.json [--verify-key key.hex]
    python scripts/skill_export.py generate-key --output key.hex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import get_db
from app.core.skill_export import (
    export_all_skills,
    generate_key,
    import_skills_from_file,
)


def cmd_export(args: argparse.Namespace) -> None:
    db = get_db()
    db.init_schema()
    skills = export_all_skills(db, private_key_path=args.sign_key)
    Path(args.output).write_text(
        json.dumps(skills, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(skills)} skill(s) to {args.output}")


def cmd_import(args: argparse.Namespace) -> None:
    db = get_db()
    db.init_schema()
    count = import_skills_from_file(args.input, db, verify_key_path=args.verify_key)
    print(f"Imported {count} skill(s) from {args.input}")


def cmd_generate_key(args: argparse.Namespace) -> None:
    key_hex = generate_key()
    Path(args.output).write_text(key_hex + "\n", encoding="utf-8")
    print(f"Generated signing key: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill import/export CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = sub.add_parser("export", help="Export all skills to JSON")
    p_export.add_argument("--output", "-o", required=True, help="Output JSON file")
    p_export.add_argument("--sign-key", help="Path to hex-encoded HMAC signing key")

    # import
    p_import = sub.add_parser("import", help="Import skills from JSON")
    p_import.add_argument("--input", "-i", required=True, help="Input JSON file")
    p_import.add_argument("--verify-key", help="Path to hex-encoded HMAC verification key")

    # generate-key
    p_keygen = sub.add_parser("generate-key", help="Generate a new HMAC signing key")
    p_keygen.add_argument("--output", "-o", required=True, help="Output key file")

    args = parser.parse_args()
    {"export": cmd_export, "import": cmd_import, "generate-key": cmd_generate_key}[args.command](args)


if __name__ == "__main__":
    main()
