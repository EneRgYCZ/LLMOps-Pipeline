"""Utility script for managing evaluation reference datasets.

Reference files map user queries to ground-truth answers, enabling
context_recall computation in the evaluation module. Matching is exact
string match on the query field — no fuzzy matching.

File format (JSON):
    [
      {"query": "What is the Eiffel Tower?",
       "reference": "A wrought-iron lattice tower in Paris, France, constructed 1887-1889."},
      ...
    ]

Usage:
    # Validate an existing references file
    python scripts/load_references.py --validate refs/experiment1.json

    # Print all loaded query/reference pairs
    python scripts/load_references.py --list refs/experiment1.json

    # Create a template references file
    python scripts/load_references.py --template refs/new_experiment.json

The wrapper loads references at startup when EVAL_REFERENCES_PATH is set.
This script is for offline dataset management only.
"""

import argparse
import json
import sys
from pathlib import Path


def validate(path: str) -> bool:
    """Validate references file format. Returns True if valid."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return False
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}", file=sys.stderr)
        return False

    if not isinstance(data, list):
        print("ERROR: Top-level structure must be a JSON array", file=sys.stderr)
        return False

    errors = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"  [{i}] not an object")
            continue
        if "query" not in item:
            errors.append(f"  [{i}] missing 'query' field")
        if "reference" not in item:
            errors.append(f"  [{i}] missing 'reference' field")
        if "query" in item and not isinstance(item["query"], str):
            errors.append(f"  [{i}] 'query' must be a string")
        if "reference" in item and not isinstance(item["reference"], str):
            errors.append(f"  [{i}] 'reference' must be a string")

    if errors:
        print(f"ERROR: Validation failed ({len(errors)} issues):", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return False

    queries = [item["query"] for item in data if "query" in item]
    duplicates = {q for q in queries if queries.count(q) > 1}
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate query strings found:")
        for d in sorted(duplicates):
            print(f"  {d!r}")

    print(f"OK: {len(data)} reference(s) loaded from {path}")
    return True


def list_refs(path: str) -> None:
    """Print all query/reference pairs."""
    with open(path) as f:
        data = json.load(f)
    for i, item in enumerate(data):
        print(f"[{i + 1}] Q: {item['query']}")
        print(f"     R: {item['reference']}")
        print()


def create_template(path: str) -> None:
    """Write a template references file."""
    out = Path(path)
    if out.exists():
        print(f"ERROR: File already exists: {path}", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = [
        {
            "query": "What is the Eiffel Tower?",
            "reference": "A wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        },
        {
            "query": "When was the Eiffel Tower built?",
            "reference": "The Eiffel Tower was constructed between 1887 and 1889.",
        },
    ]
    out.write_text(json.dumps(template, indent=2) + "\n")
    print(f"Template written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage evaluation reference datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", metavar="FILE", help="Validate a references JSON file")
    group.add_argument("--list", metavar="FILE", help="List all query/reference pairs")
    group.add_argument("--template", metavar="FILE", help="Create a template references file")
    args = parser.parse_args()

    if args.validate:
        ok = validate(args.validate)
        sys.exit(0 if ok else 1)
    elif args.list:
        list_refs(args.list)
    elif args.template:
        create_template(args.template)


if __name__ == "__main__":
    main()
