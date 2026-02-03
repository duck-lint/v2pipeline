from __future__ import annotations

import argparse
from pathlib import Path

_WARNED: set[str] = set()

def _warn_deprecated(flag: str, replacement: str) -> None:
    if flag in _WARNED:
        return
    _WARNED.add(flag)
    print(f"[deprecation] {flag} is deprecated; use {replacement}")


def iter_jsonl_files(root: Path, recursive: bool, output_path: Path) -> list[Path]:
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(
        [
            p
            for p in root.glob(pattern)
            if p.is_file()
            and p.name.endswith(".chunks.jsonl")
            and p.resolve() != output_path
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks_dir",
        type=str,
        help="Folder containing per-file JSONL chunks",
    )
    ap.add_argument("--chunks_path", type=str, help=argparse.SUPPRESS)
    ap.add_argument(
        "--output_jsonl",
        type=str,
        help="Output JSONL path",
        required=True,
    )
    ap.add_argument(
        "--no_recursive",
        action="store_true",
        help="Do not recurse into subfolders",
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    chunks_dir = args.chunks_dir
    if args.chunks_path:
        _warn_deprecated("--chunks_path", "--chunks_dir")
        if args.chunks_dir:
            raise ValueError("Use only one of --chunks_dir or --chunks_path")
        chunks_dir = args.chunks_path
    args.chunks_dir = chunks_dir

    if not chunks_dir:
        raise ValueError("--chunks_dir is required")

    chunks_dir_path = Path(chunks_dir).resolve()
    if not chunks_dir_path.exists():
        raise FileNotFoundError(f"Missing chunks_dir: {chunks_dir_path}")
    if not chunks_dir_path.is_dir():
        raise NotADirectoryError(f"chunks_dir must be a directory: {chunks_dir_path}")

    output_path = Path(args.output_jsonl).resolve()

    files = iter_jsonl_files(chunks_dir_path, recursive=not args.no_recursive, output_path=output_path)
    if not files:
        print("[merge_chunks] no JSONL files found")
        return 0

    print(f"[merge_chunks] chunks_dir={chunks_dir_path}")
    print(f"[merge_chunks] files={len(files)}")
    print(f"[merge_chunks] output_jsonl={output_path}")

    if args.dry_run:
        print("[merge_chunks] dry_run=True (no write performed)")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
