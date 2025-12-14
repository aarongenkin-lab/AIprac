"""
Build a small WikiTableQuestions mini-set (JSONL) with tables flattened to text.
Expects WTQ data locally (no network).
"""

import argparse
import json
import re
import csv
from pathlib import Path
from typing import List


def csv_to_text(csv_path: Path, max_rows: int | None = None) -> str:
    """Flatten CSV to a TSV-like string."""
    lines: List[str] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            cleaned = [re.sub(r"\s+", " ", cell).strip() for cell in row]
            lines.append("\t".join(cleaned))
    return "\n".join(lines)


def parse_tsv(tsv_path: Path) -> List[dict]:
    """Parse a WTQ tsv line-by-line into dicts; expects tab-separated columns:
    id, question, table_id, answer
    """
    rows = []
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            ex_id, question, table_id, answer = parts[0], parts[1], parts[2], parts[3]
            rows.append(
                {
                    "id": ex_id,
                    "question": question,
                    "table_id": table_id,
                    "answer": answer,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wtq-root", required=True, help="Path to WTQ data directory (contains random-split TSV files)")
    ap.add_argument("--split", default="random-split-1-dev.tsv", help="TSV filename (e.g., random-split-1-dev.tsv)")
    ap.add_argument("--max", type=int, default=20, help="Max examples to export")
    ap.add_argument("--max-rows", type=int, default=30, help="Max rows per table to keep (flattened)")
    ap.add_argument("--out", default="benchmarks/wtq_mini/wtq_mini.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    wtq_root = Path(args.wtq_root)
    tsv_path = wtq_root / args.split
    csv_root = wtq_root.parent  # table_id paths are relative to repo root (e.g., csv/204-csv/772.csv)

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    rows = parse_tsv(tsv_path)[: args.max]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        kept = 0
        for row in rows:
            table_id = row["table_id"]
            table_path = csv_root / table_id
            if not table_path.exists():
                continue
            table_text = csv_to_text(table_path, max_rows=args.max_rows)
            rec = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "table_id": table_id,
                "table_text": table_text,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()
