from pathlib import Path
import argparse
from common import (
    read_text,
    strip_yaml_frontmatter,
    normalize_markdown_light,
    replace_wikilinks_and_collect,
    parse_yaml_frontmatter,
    parse_source_date,
    write_text,
    sha256_text,
    parse_date_field,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_md", type=str, required=True, help="Path to stage_0_raw/*.md")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean", help="default=stage_1_clean")
    ap.add_argument("--emit_links", action="store_true", help="Write out_links JSON next to cleaned text")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"[stage_1_clean] args: {args}")

    src = Path(args.stage0_md).resolve()
    raw = read_text(src)
    body, yaml_block = strip_yaml_frontmatter(raw)
    meta = parse_yaml_frontmatter(yaml_block or "")
    entry_date = parse_date_field(meta, "journal_entry_date")
    source_date = parse_date_field(meta, "note_creation_date")

    print(f"[stage_1_clean] src={src}")

    normalized = normalize_markdown_light(body)
    replaced, out_links = replace_wikilinks_and_collect(normalized)

    stage1 = Path(args.stage1_dir).resolve()
    stage1.mkdir(parents=True, exist_ok=True)

    out_txt = stage1 / f"{src.stem}.clean.txt"
    out_links_json = stage1 / f"{src.stem}.out_links.json"

    print("[stage_1] ---- summary ----")
    print(f"[stage_1] file={src.name}")
    print(f"[stage_1] yaml_present={'yes' if yaml_block else 'no'} | yaml_keys={len(meta)} | entry_date={entry_date} | source_date={source_date}")
    print(f"[stage_1] chars_raw={len(raw)} | chars_body={len(body)} | chars_clean={len(replaced)}")
    print(f"[stage_1] out_links={len(out_links)}")
    print(f"[stage_1] clean_hash={sha256_text(replaced)[:12]}…")
    print("[stage_1] preview:")
    print(replaced[:300].replace("\n", "\\n"))

    if args.dry_run:
        print("[stage_1] dry_run=True (no write performed)")
        return

    write_text(out_txt, replaced)
    print(f"[stage_1] wrote: {out_txt}")

    if args.emit_links:
        import json
        out_links_json.write_text(json.dumps(out_links, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stage_1] wrote: {out_links_json}")

if __name__ == "__main__":
    main()
