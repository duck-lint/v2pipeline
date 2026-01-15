from pathlib import Path
import argparse
import re
from common import (
    read_text,
    strip_yaml_frontmatter,
    parse_yaml_frontmatter,
    normalize_markdown_light,
    replace_wikilinks_and_collect,
    sha256_bytes,
    sha256_text,
    write_jsonl,
    split_into_sections,
    parse_date_field,
)

CHUNKER_VERSION = "v0.1"


def iter_md_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def build_chunks(src: Path, stage0_root: Path, args: argparse.Namespace) -> tuple[list[dict], Path, str]:
    raw_bytes = src.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    body, yaml_block = strip_yaml_frontmatter(raw_text)
    meta = parse_yaml_frontmatter(yaml_block or "")

    if stage0_root.is_dir():
        rel = src.relative_to(stage0_root)
    else:
        rel = Path(src.name)

    stage1_path = Path(args.stage1_dir).resolve() / rel
    stage1_path = stage1_path.with_suffix(".clean.txt")

    if args.prefer_stage1:
        if not stage1_path.exists():
            raise FileNotFoundError(f"--prefer_stage1 set but stage1 file missing: {stage1_path}")
        chunk_input = read_text(stage1_path)
    else:
        chunk_input = normalize_markdown_light(body)

    doc_id = str(meta.get("uuid") or "").strip() or sha256_bytes(raw_bytes)[:24]  # V1 fallback
    rel_path = str(rel)
    entry_date = parse_date_field(meta, "journal_entry_date")
    source_date = parse_date_field(meta, "note_creation_date")
    source_hash = sha256_bytes(raw_bytes)

    sections = split_into_sections(chunk_input)

    rows = []
    total_chunks = 0

    for anchor, section_title, section_raw in sections:
        # normalize then replace links per chunk
        normalized = normalize_markdown_light(section_raw)

        # Split into paragraphs (blank-line separated)
        paragraphs_raw = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        # Drop empty sections (this also drops "header-only" sections because header lines are no longer in section_raw)
        if not paragraphs_raw:
            continue

        parts: list[str] = []
        parts_links: list[list[dict]] = []

        for para_raw in paragraphs_raw:
            if len(para_raw) <= args.max_chars:
                chunk_text, chunk_links = replace_wikilinks_and_collect(para_raw)
                parts.append(chunk_text.strip() + "\n")
                parts_links.append(chunk_links)
            else:
                # deterministic split for huge paragraphs
                buf: list[str] = []
                cur = 0
                for piece in re.split(r"(?<=[.!?])\s+", para_raw):
                    if cur + len(piece) + 1 > args.max_chars and buf:
                        combined_raw = " ".join(buf)
                        chunk_text, chunk_links = replace_wikilinks_and_collect(combined_raw)
                        parts.append(chunk_text.strip() + "\n")
                        parts_links.append(chunk_links)
                        buf = []
                        cur = 0
                    buf.append(piece)
                    cur += len(piece) + 1

                if buf:
                    combined_raw = " ".join(buf)
                    chunk_text, chunk_links = replace_wikilinks_and_collect(combined_raw)
                    parts.append(chunk_text.strip() + "\n")
                    parts_links.append(chunk_links)

        for idx, chunk_text in enumerate(parts):
            chunk_id = f"{doc_id}::{anchor}::{idx}"
            content_hash = sha256_text(chunk_text)

            rows.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_anchor": anchor,
                    "chunk_title": section_title,
                    "chunk_index": idx,
                    "rel_path": rel_path,
                    "entry_date": entry_date,
                    "source_date": source_date,
                    "source_hash": source_hash,
                    "content_hash": content_hash,
                    "embed_model": args.embed_model,
                    "embed_dim": args.embed_dim,
                    "chunker_version": CHUNKER_VERSION,
                    "out_links": parts_links[idx],
                }
            })
            total_chunks += 1

    if args.out_jsonl and not stage0_root.is_dir():
        out_path = Path(args.out_jsonl).resolve()
    else:
        out_path = Path(args.out_dir).resolve() / rel
        out_path = out_path.with_suffix(".chunks.jsonl")

    summary = f"sections={len(sections)} | chunks={total_chunks}"
    return rows, out_path, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_path", type=str, help="Path to stage_0_raw file or folder")
    ap.add_argument("--stage0_raw", type=str, help="(deprecated) Path to stage_0_raw/*.md")
    ap.add_argument("--out_jsonl", type=str, help="Output JSONL (single-file mode)")
    ap.add_argument("--out_dir", type=str, default="stage_2_chunks", help="Output folder for per-file JSONL")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed_dim", type=int, default=384, help="default=384")
    ap.add_argument("--max_chars", type=int, default=2500, help="Split section if longer than this (V1 safety)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean")
    ap.add_argument("--prefer_stage1", action="store_true", help="Chunk from stage_1_clean if available")
    ap.add_argument("--no_recursive", action="store_true", help="If stage0_path is a folder, do not recurse")
    args = ap.parse_args()

    src_arg = args.stage0_path or args.stage0_raw
    if not src_arg:
        raise ValueError("Provide --stage0_path (file or folder) or --stage0_raw (deprecated).")

    stage0_root = Path(src_arg).resolve()
    print(f"[stage_02_chunk] args: {args}")

    if stage0_root.is_dir() and args.out_jsonl:
        raise ValueError("--out_jsonl is only valid for single-file input")

    if stage0_root.is_dir():
        files = iter_md_files(stage0_root, recursive=not args.no_recursive)
    else:
        files = [stage0_root]

    if not files:
        print("[stage_2] no markdown files found")
        return

    for src in files:
        print(f"[stage_02_chunk] src={src}")
        rows, out_path, summary = build_chunks(src, stage0_root, args)

        print("[stage_2] ---- summary ----")
        print(f"[stage_2] file={src.name}")
        print(f"[stage_2] {summary}")
        print(f"[stage_2] out_jsonl={out_path}")
        if rows:
            print("[stage_2] first_chunk_preview:")
            print(rows[0]["text"][:220].replace("\n", "\\n"))

        if args.dry_run:
            print("[stage_2] dry_run=True (no write performed)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_path, rows)
        print("[stage_2] wrote jsonl")


if __name__ == "__main__":
    main()
