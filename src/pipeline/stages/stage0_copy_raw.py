from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline.common import iter_markdown_files

_WARNED: set[str] = set()

def _warn_deprecated(flag: str, replacement: str) -> None:
    if flag in _WARNED:
        return
    _WARNED.add(flag)
    print(f"[deprecation] {flag} is deprecated; use {replacement}")

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True, help="Path to a markdown file or a folder")
    ap.add_argument("--stage0_dir", type=str, default="stage_0_raw", help="Output folder for raw note copy")
    ap.add_argument("--stage0_path", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--no_recursive", action="store_true", help="If input_path is a folder, do not recurse")
    ap.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable)")
    ap.add_argument("--dry_run", action="store_true", help="Print what would happen; do not copy")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    stage0_dir = args.stage0_dir
    if args.stage0_path:
        _warn_deprecated("--stage0_path", "--stage0_dir")
        if args.stage0_dir != "stage_0_raw":
            raise ValueError("Use only one of --stage0_dir or --stage0_path")
        stage0_dir = args.stage0_path
    args.stage0_dir = stage0_dir

    src_root = Path(args.input_path).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Not found: {src_root}")
    if not src_root.is_dir() and not src_root.is_file():
        raise ValueError(f"input_path must be a file or directory: {src_root}")

    recursive = not args.no_recursive
    files = iter_markdown_files(src_root, recursive, args.exclude, exclude_hidden=True)
    if not files:
        print("[stage_0] no markdown files found")
        return 0

    dst_root = Path(stage0_dir).expanduser().resolve()

    print("[stage_0] copy markdown file(s) to stage_0_raw/")
    print(f"[stage_0] args: {args}")
    print(f"[stage_0] src_root={src_root}")
    print(f"[stage_0] dst_root={dst_root}")
    print(f"[stage_0] files={len(files)}")

    if not args.dry_run:
        if dst_root.exists() and not dst_root.is_dir():
            raise NotADirectoryError(f"stage0_dir must be a directory: {dst_root}")
        dst_root.mkdir(parents=True, exist_ok=True)

    for src in files:
        if src_root.is_dir():
            rel = src.relative_to(src_root)
        else:
            rel = src.name
        dst = dst_root / rel

        # If user already put the file in stage_0_raw, don't crash.
        try:
            if src.resolve() == dst.resolve():
                print(f"[stage_0] NOTE: already in stage_0_raw, skipping: {src}")
                continue
        except FileNotFoundError:
            pass

        if args.dry_run:
            print(f"[stage_0] dry_run=True would copy: {src} -> {dst}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if not args.dry_run:
        print("[stage_0] copied")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
