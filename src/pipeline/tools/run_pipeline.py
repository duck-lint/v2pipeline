from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, List

from pipeline.stages import stage0_copy_raw, stage1_clean, stage2_chunk, stage3_chroma
from pipeline.tools import merge_chunks_jsonl


def run_cmd(label: str, main_func: Callable[[list[str] | None], int], argv: list[str]) -> None:
    print(f"[run_pipeline] cmd: {label} {' '.join(argv)}")
    exit_code = main_func(argv)
    if exit_code not in (0, None):
        raise SystemExit(exit_code)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True, help="Input markdown file or folder")
    ap.add_argument("--stage0_dir", type=str, default="stage_0_raw")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean")
    ap.add_argument("--stage2_dir", type=str, default="stage_2_chunks")
    ap.add_argument("--prefer_stage1", action="store_true", help="Use stage_1_clean for chunking")
    ap.add_argument("--no_recursive", action="store_true", help="Do not recurse into subfolders")
    ap.add_argument("--emit_links", action="store_true")
    ap.add_argument("--clean_stage0", action="store_true", help="Delete stage_0_raw before copying")
    ap.add_argument("--clean_stage1", action="store_true", help="Delete stage_1_clean before cleaning")
    ap.add_argument("--clean_stage2", action="store_true", help="Delete stage_2_chunks before writing new chunks")
    ap.add_argument("--merge_chunks", action="store_true")
    ap.add_argument("--merged_jsonl", type=str, default="stage_2_chunks_merged.jsonl")
    ap.add_argument("--build_chroma", action="store_true")
    ap.add_argument("--chunks_jsonl", type=str, help="Explicit JSONL for stage 3 (overrides merged_jsonl)")
    ap.add_argument(
        "--chunks_dir",
        type=str,
        help="Batch mode: build from all JSONL files under this directory",
    )
    ap.add_argument(
        "--chunks_pattern",
        type=str,
        default="*.jsonl",
        help="Pattern for --chunks_dir (default=*.jsonl)",
    )
    ap.add_argument(
        "--chunks_recursive",
        action="store_true",
        help="Recurse when using --chunks_dir",
    )
    ap.add_argument("--persist_dir", type=str, default="stage_3_chroma")
    ap.add_argument("--collection", type=str, default="v1_chunks")
    ap.add_argument("--mode", type=str, choices=["rebuild", "append", "upsert"], default="upsert")
    ap.add_argument("--upsert", action="store_true", help="Alias for --mode upsert")
    ap.add_argument("--skip_unchanged", action="store_true", help="When upserting, skip chunks whose hash hasn't changed")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--top_to_bottom", action="store_true", help="Run stages 0->3 in order (default behavior)")
    return ap


def _add_flag(argv: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.upsert:
        args.mode = "upsert"

    if args.clean_stage0:
        stage0_dir = Path(args.stage0_dir).resolve()
        if stage0_dir.exists():
            print(f"[run_pipeline] removing stage0_dir: {stage0_dir}")
            shutil.rmtree(stage0_dir)

    stage0_args = [
        "--input_path",
        args.input_path,
        "--stage0_dir",
        args.stage0_dir,
    ]
    _add_flag(stage0_args, "--no_recursive", args.no_recursive)
    _add_flag(stage0_args, "--dry_run", args.dry_run)
    run_cmd("python -m pipeline stage0", stage0_copy_raw.main, stage0_args)

    if args.clean_stage1:
        stage1_dir = Path(args.stage1_dir).resolve()
        if stage1_dir.exists():
            print(f"[run_pipeline] removing stage1_dir: {stage1_dir}")
            shutil.rmtree(stage1_dir)

    stage1_args = [
        "--stage0_path",
        args.stage0_dir,
        "--stage1_dir",
        args.stage1_dir,
    ]
    _add_flag(stage1_args, "--no_recursive", args.no_recursive)
    _add_flag(stage1_args, "--dry_run", args.dry_run)
    _add_flag(stage1_args, "--emit_links", args.emit_links)
    run_cmd("python -m pipeline stage1", stage1_clean.main, stage1_args)

    if args.clean_stage2:
        stage2_dir = Path(args.stage2_dir).resolve()
        if stage2_dir.exists():
            print(f"[run_pipeline] removing stage2_dir: {stage2_dir}")
            shutil.rmtree(stage2_dir)

    stage2_args = [
        "--stage0_path",
        args.stage0_dir,
        "--stage1_dir",
        args.stage1_dir,
        "--out_dir",
        args.stage2_dir,
    ]
    _add_flag(stage2_args, "--no_recursive", args.no_recursive)
    _add_flag(stage2_args, "--dry_run", args.dry_run)
    _add_flag(stage2_args, "--prefer_stage1", args.prefer_stage1)
    run_cmd("python -m pipeline stage2", stage2_chunk.main, stage2_args)

    merged_jsonl = None
    if args.merge_chunks:
        merge_args = [
            "--chunks_dir",
            args.stage2_dir,
            "--output_jsonl",
            args.merged_jsonl,
        ]
        _add_flag(merge_args, "--no_recursive", args.no_recursive)
        _add_flag(merge_args, "--dry_run", args.dry_run)
        run_cmd("python -m pipeline merge", merge_chunks_jsonl.main, merge_args)
        merged_jsonl = args.merged_jsonl

    if args.build_chroma:
        if args.chunks_dir:
            chunks_dir = Path(args.chunks_dir).resolve()
            if not chunks_dir.exists():
                raise FileNotFoundError(f"Missing chunks_dir: {chunks_dir}")
            if not chunks_dir.is_dir():
                raise NotADirectoryError(f"chunks_dir must be a directory: {chunks_dir}")
            pattern = f"**/{args.chunks_pattern}" if args.chunks_recursive else args.chunks_pattern
            files = sorted([p for p in chunks_dir.glob(pattern) if p.is_file()])
            if not files:
                raise FileNotFoundError(f"No JSONL files found in: {chunks_dir}")
            for path in files:
                stage3_args = [
                    "--chunks_jsonl",
                    str(path),
                    "--persist_dir",
                    args.persist_dir,
                    "--collection",
                    args.collection,
                    "--mode",
                    args.mode,
                ]
                _add_flag(stage3_args, "--dry_run", args.dry_run)
                _add_flag(stage3_args, "--skip_unchanged", args.skip_unchanged)
                run_cmd("python -m pipeline stage3", stage3_chroma.main, stage3_args)
        else:
            chunks_input = args.chunks_jsonl or merged_jsonl or args.merged_jsonl
            if not args.dry_run and not Path(chunks_input).exists():
                raise FileNotFoundError(
                    "Missing chunks JSONL for stage 3. "
                    "Run with --merge_chunks or provide --chunks_jsonl."
                )
            stage3_args = [
                "--chunks_jsonl",
                chunks_input,
                "--persist_dir",
                args.persist_dir,
                "--collection",
                args.collection,
                "--mode",
                args.mode,
            ]
            _add_flag(stage3_args, "--dry_run", args.dry_run)
            _add_flag(stage3_args, "--skip_unchanged", args.skip_unchanged)
            run_cmd("python -m pipeline stage3", stage3_chroma.main, stage3_args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
