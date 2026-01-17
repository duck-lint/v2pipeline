from pathlib import Path
import argparse
import sys
from common import (
    configure_stdout,
    extract_wikilinks,
    iter_markdown_files,
    read_text,
    strip_yaml_frontmatter,
    normalize_markdown_light,
    parse_yaml_frontmatter,
    write_text,
    sha256_text,
    parse_date_field,
)

def _console_safe(s: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")

def process_file(src: Path, stage1_root: Path, rel_path: Path, args: argparse.Namespace) -> None:
    raw = read_text(src)
    body, yaml_block = strip_yaml_frontmatter(raw)
    yaml_error = None
    try:
        meta = parse_yaml_frontmatter(yaml_block or "")
    except Exception as exc:
        if args.yaml_mode == "strict":
            raise
        yaml_error = f"{type(exc).__name__}: {exc}"
        meta = {}
    entry_date = parse_date_field(meta, "journal_entry_date")
    source_date = parse_date_field(meta, "note_creation_date")

    normalized = normalize_markdown_light(body)
    cleaned_text = normalized
    out_links = extract_wikilinks(cleaned_text) if args.emit_links else []

    out_txt = stage1_root / rel_path
    out_txt = out_txt.with_suffix(".clean.txt")
    out_links_json = out_txt.with_suffix(".out_links.json")

    print("[stage_1] ---- summary ----")
    print(f"[stage_1] file={src.name}")
    print(f"[stage_1] yaml_present={'yes' if yaml_block else 'no'} | yaml_keys={len(meta)} | entry_date={entry_date} | source_date={source_date}")
    if yaml_error:
        print(f"[stage_1] yaml_error={yaml_error}")
    print(f"[stage_1] chars_raw={len(raw)} | chars_body={len(body)} | chars_clean={len(cleaned_text)}")
    print(f"[stage_1] out_links={len(out_links)}")
    print(f"[stage_1] clean_hash={sha256_text(cleaned_text)[:12]}")
    print("[stage_1] preview:")
    print(_console_safe(cleaned_text[:300].replace("\n", "\\n")))

    if args.dry_run:
        print("[stage_1] dry_run=True (no write performed)")
        return

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    write_text(out_txt, cleaned_text)
    print(f"[stage_1] wrote: {out_txt}")

    if args.emit_links:
        import json
        out_links_json.write_text(json.dumps(out_links, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stage_1] wrote: {out_links_json}")


def main() -> None:
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_path", type=str, required=True, help="Path to stage_0_raw file or folder")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean", help="default=stage_1_clean")
    ap.add_argument("--emit_links", action="store_true", help="Write out_links JSON next to cleaned text")
    ap.add_argument("--no_recursive", action="store_true", help="If stage0_path is a folder, do not recurse")
    ap.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable)")
    ap.add_argument("--yaml_mode", type=str, choices=["strict", "lenient"], default="strict")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.stage0_path).resolve()
    stage1_root = Path(args.stage1_dir).resolve()

    print(f"[stage_1_clean] args: {args}")

    if src_root.is_dir():
        files = iter_markdown_files(src_root, recursive=not args.no_recursive, exclude_globs=args.exclude, exclude_hidden=True)
    else:
        files = iter_markdown_files(src_root, recursive=False, exclude_globs=args.exclude, exclude_hidden=True)

    if not files:
        print("[stage_1] no markdown files found")
        return

    for src in files:
        rel = src.relative_to(src_root) if src_root.is_dir() else Path(src.name)
        process_file(src, stage1_root, rel, args)


if __name__ == "__main__":
    main()
