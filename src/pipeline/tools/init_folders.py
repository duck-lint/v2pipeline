from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root")
    ap.add_argument("--dry_run", action="store_true", help="Print what would happen; do not create")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    print(f"[stage_folders] args: {args}")

    root = Path(args.root).resolve()
    targets = [
        root / "stage_0_raw",
        root / "stage_1_clean",
        root / "stage_2_chunks",
        root / "stage_3_chroma",
    ]

    if args.dry_run:
        print("[stage_folders] dry_run=True (no write performed)")
    else:
        for path in targets:
            ensure_dir(path)

    print("Created/verified folders:")
    for path in targets:
        print(f" - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
