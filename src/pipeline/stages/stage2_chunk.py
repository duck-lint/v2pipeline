from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from pipeline.common import (
    canonicalize_heading_path,
    configure_stdout,
    generate_chunk_identity,
    iter_markdown_files,
    normalize_markdown_light,
    parse_date_field,
    parse_yaml_frontmatter,
    read_text,
    replace_wikilinks_and_collect,
    sha256_bytes,
    sha256_text,
    split_into_sections,
    stable_doc_id_from_rel_path,
    strip_yaml_frontmatter,
    write_jsonl,
)

CHUNKER_VERSION = "v0.1"

_WARNED: set[str] = set()

def _warn_deprecated(flag: str, replacement: str) -> None:
    if flag in _WARNED:
        return
    _WARNED.add(flag)
    print(f"[deprecation] {flag} is deprecated; use {replacement}")

def _infer_stage0_root_for_file(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "stage_0_raw":
            return parent
    return path.parent

def _console_safe(s: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")


def build_chunks(src: Path, stage0_root: Path, args: argparse.Namespace) -> tuple[list[dict], Path, str]:
    raw_bytes = src.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    body, yaml_block = strip_yaml_frontmatter(raw_text)
    yaml_error = None
    try:
        meta = parse_yaml_frontmatter(yaml_block or "")
    except Exception as exc:
        if args.yaml_mode == "strict":
            raise
        yaml_error = f"{type(exc).__name__}: {exc}"
        meta = {}

    if stage0_root.is_dir():
        rel = src.relative_to(stage0_root)
    else:
        rel_root = _infer_stage0_root_for_file(src)
        rel = src.relative_to(rel_root) if rel_root in src.parents else Path(src.name)

    stage1_path = Path(args.stage1_dir).resolve() / rel
    stage1_path = stage1_path.with_suffix(".clean.txt")

    if args.prefer_stage1 and stage1_path.exists():
        chunk_input = read_text(stage1_path)
    else:
        if args.prefer_stage1 and not stage1_path.exists():
            print(f"[stage_2] warning: prefer_stage1 set but missing: {stage1_path} (falling back to stage0)")
        chunk_input = normalize_markdown_light(body)

    doc_id = str(meta.get("uuid") or "").strip() or stable_doc_id_from_rel_path(rel)
    rel_path = str(rel)
    source_uri = rel_path.replace("\\", "/")
    entry_date = parse_date_field(meta, "journal_entry_date")
    source_hash = sha256_bytes(raw_bytes)
    try:
        mtime_ts = src.stat().st_mtime
        source_date = datetime.fromtimestamp(mtime_ts).date().isoformat()
    except OSError:
        source_date = None
    parts = [p for p in rel.parts if p not in (".", "")]
    folder = parts[0] if parts else ""
    doc_type = str(meta.get("doc_type") or "").strip() or (folder.lower() if folder else "note")
    sensitivity = str(meta.get("sensitivity") or "").strip() or "private"

    sections = split_into_sections(chunk_input)
    heading_counts = Counter(
        " > ".join(canonicalize_heading_path(heading_path)) for _, _, heading_path, _ in sections
    )
    heading_seen: defaultdict[str, int] = defaultdict(int)

    rows = []
    total_chunks = 0

    for anchor, section_title, heading_path, section_raw in sections:
        canon_heading_str = " > ".join(canonicalize_heading_path(heading_path))
        ordinal = heading_seen[canon_heading_str]
        heading_seen[canon_heading_str] += 1
        section_ordinal = ordinal if heading_counts[canon_heading_str] > 1 else None
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
            identity = generate_chunk_identity(
                source_uri,
                heading_path,
                idx,
                chunk_text,
                section_ordinal=section_ordinal,
            )
            chunk_id = identity["chunk_id"]
            chunk_key = identity["chunk_key"]
            chunk_hash = identity["chunk_hash"]
            content_hash = sha256_text(chunk_text)

            rows.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_key": chunk_key,
                        "chunk_hash": chunk_hash,
                        "chunk_anchor": anchor,
                        "chunk_title": section_title,
                        "heading_path": identity["heading_path"],
                        "chunk_index": idx,
                        "rel_path": rel_path,
                        "source_uri": identity["source_uri"],
                        "cleaned_text": chunk_text,
                        "entry_date": entry_date,
                        "source_date": source_date,
                        "source_hash": source_hash,
                        "content_hash": content_hash,
                        "folder": folder,
                        "doc_type": doc_type,
                        "sensitivity": sensitivity,
                        "chunker_version": CHUNKER_VERSION,
                        "out_links": parts_links[idx],
                        **({"yaml_error": yaml_error} if yaml_error else {}),
                    },
                }
            )
            total_chunks += 1

    out_path = Path(args.out_dir).resolve() / rel
    out_path = out_path.with_suffix(".chunks.jsonl")

    summary = f"sections={len(sections)} | chunks={total_chunks}"
    return rows, out_path, summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_path", type=str, required=True, help="Path to stage_0_raw file or folder")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean")
    ap.add_argument("--out_dir", type=str, default="stage_2_chunks", help="Output folder for per-file JSONL")
    ap.add_argument("--stage1_path", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--out_path", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--max_chars", type=int, default=2500, help="Split section if longer than this (V1 safety)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--prefer_stage1", action="store_true", help="Chunk from stage_1_clean if available")
    ap.add_argument("--no_recursive", action="store_true", help="If stage0_path is a folder, do not recurse")
    ap.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable)")
    ap.add_argument("--yaml_mode", type=str, choices=["strict", "lenient"], default="strict")
    return ap


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    ap = build_parser()
    args = ap.parse_args(argv)

    stage1_dir = args.stage1_dir
    if args.stage1_path:
        _warn_deprecated("--stage1_path", "--stage1_dir")
        if args.stage1_dir != "stage_1_clean":
            raise ValueError("Use only one of --stage1_dir or --stage1_path")
        stage1_dir = args.stage1_path

    out_dir = args.out_dir
    if args.out_path:
        _warn_deprecated("--out_path", "--out_dir")
        if args.out_dir != "stage_2_chunks":
            raise ValueError("Use only one of --out_dir or --out_path")
        out_dir = args.out_path

    args.stage1_dir = stage1_dir
    args.out_dir = out_dir

    stage0_root = Path(args.stage0_path).resolve()
    print(f"[stage_2] args: {args}")
    if not stage0_root.exists():
        raise FileNotFoundError(f"Missing stage0_path: {stage0_root}")
    if stage0_root.exists() and not stage0_root.is_dir() and not stage0_root.is_file():
        raise ValueError(f"stage0_path must be a file or directory: {stage0_root}")

    stage1_root = Path(stage1_dir).resolve()
    if not stage1_root.exists() or not stage1_root.is_dir():
        if args.dry_run:
            print(f"[stage_2] dry_run=True allowing missing stage1_dir: {stage1_root}")
        else:
            raise NotADirectoryError(f"stage1_dir must be a directory: {stage1_root}")

    out_root = Path(out_dir).resolve()
    if out_root.exists() and not out_root.is_dir():
        raise NotADirectoryError(f"out_dir must be a directory: {out_root}")

    if stage0_root.is_dir():
        files = iter_markdown_files(
            stage0_root,
            recursive=not args.no_recursive,
            exclude_globs=args.exclude,
            exclude_hidden=True,
        )
    else:
        files = iter_markdown_files(
            stage0_root,
            recursive=False,
            exclude_globs=args.exclude,
            exclude_hidden=True,
        )

    if not files:
        print("[stage_2] no markdown files found")
        return 0

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    for src in files:
        print(f"[stage_2] src={src}")
        rows, out_path, summary = build_chunks(src, stage0_root, args)

        print("[stage_2] ---- summary ----")
        print(f"[stage_2] file={src.name}")
        print(f"[stage_2] {summary}")
        print(f"[stage_2] out_path={out_path}")
        if rows:
            print("[stage_2] first_chunk_preview:")
            print(_console_safe(rows[0]["text"][:220].replace("\n", "\\n")))

        if args.dry_run:
            print("[stage_2] dry_run=True (no write performed)")
            continue

        seen_ids: set[str] = set()
        for row in rows:
            cid = row.get("metadata", {}).get("chunk_id")
            if not cid:
                continue
            if cid in seen_ids:
                meta = row.get("metadata", {})
                raise ValueError(
                    "Duplicate chunk_id in output rows: "
                    f"{cid} | source_uri={meta.get('source_uri')} | "
                    f"chunk_anchor={meta.get('chunk_anchor')} | "
                    f"chunk_title={meta.get('chunk_title')} | "
                    f"chunk_index={meta.get('chunk_index')}"
                )
            seen_ids.add(cid)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_path, rows)
        print("[stage_2] wrote jsonl")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
