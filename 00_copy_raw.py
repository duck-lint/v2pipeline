from pathlib import Path
import argparse
import shutil

from common import iter_markdown_files

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True, help="Path to a markdown file or a folder")
    ap.add_argument("--stage0_dir", type=str, default="stage_0_raw", help="Output folder for raw note copy")
    ap.add_argument("--no_recursive", action="store_true", help="If input_path is a folder, do not recurse")
    ap.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable)")
    ap.add_argument("--dry_run", action="store_true", help="Print what would happen; do not copy")
    args = ap.parse_args()

    src_root = Path(args.input_path).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Not found: {src_root}")

    recursive = not args.no_recursive
    files = iter_markdown_files(src_root, recursive, args.exclude, exclude_hidden=True)
    if not files:
        print("[stage_0] no markdown files found")
        return

    dst_root = Path(args.stage0_dir).expanduser().resolve()

    print("[stage_0] copy markdown file(s) to stage_0_raw/")
    print(f"[stage_0] args: {args}")
    print(f"[stage_0] src_root={src_root}")
    print(f"[stage_0] dst_root={dst_root}")
    print(f"[stage_0] files={len(files)}")

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


if __name__ == "__main__":
    main()
