from __future__ import annotations

import argparse
from pathlib import Path


def iter_jsonl_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks_dir",
        type=str,
        default="stage_2_chunks",
        help="Folder containing per-file JSONL chunks",
    )
    ap.add_argument(
        "--output_jsonl",
        type=str,
        default="stage_2_chunks_merged.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--no_recursive",
        action="store_true",
        help="Do not recurse into subfolders",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir).resolve()
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Missing chunks_dir: {chunks_dir}")
    if not chunks_dir.is_dir():
        raise NotADirectoryError(f"chunks_dir must be a directory: {chunks_dir}")

    files = iter_jsonl_files(chunks_dir, recursive=not args.no_recursive)
    if not files:
        print("[merge_chunks] no JSONL files found")
        return

    output_path = Path(args.output_jsonl).resolve()

    print(f"[merge_chunks] chunks_dir={chunks_dir}")
    print(f"[merge_chunks] files={len(files)}")
    print(f"[merge_chunks] output_jsonl={output_path}")

    if args.dry_run:
        print("[merge_chunks] dry_run=True (no write performed)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
        for path in files:
            with path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    out_f.write(line + "\n")

    print("[merge_chunks] wrote merged JSONL")


if __name__ == "__main__":
    main()
