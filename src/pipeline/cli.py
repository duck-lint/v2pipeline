from __future__ import annotations

from typing import Callable

import typer

from pipeline.stages import stage0_copy_raw, stage1_clean, stage2_chunk, stage3_chroma
from pipeline.tools import init_folders, merge_chunks_jsonl, query, run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _forward(ctx: typer.Context, main_func: Callable[[list[str] | None], int]) -> None:
    argv = list(ctx.args)
    raise SystemExit(main_func(argv))


@app.command(
    "init-folders",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def init_folders_cmd(ctx: typer.Context) -> None:
    """Initialize stage folders (delegates to init_folders.py)."""
    _forward(ctx, init_folders.main)


@app.command(
    "stage0",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def stage0_cmd(ctx: typer.Context) -> None:
    """Stage 0: copy raw inputs (delegates to 00_copy_raw.py)."""
    _forward(ctx, stage0_copy_raw.main)


@app.command(
    "stage1",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def stage1_cmd(ctx: typer.Context) -> None:
    """Stage 1: clean text (delegates to 01_clean.py)."""
    _forward(ctx, stage1_clean.main)


@app.command(
    "stage2",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def stage2_cmd(ctx: typer.Context) -> None:
    """Stage 2: chunk into JSONL (delegates to 02_chunk.py)."""
    _forward(ctx, stage2_chunk.main)


@app.command(
    "merge",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def merge_cmd(ctx: typer.Context) -> None:
    """Merge per-file chunks JSONL (delegates to merge_chunks_jsonl.py)."""
    _forward(ctx, merge_chunks_jsonl.main)


@app.command(
    "stage3",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def stage3_cmd(ctx: typer.Context) -> None:
    """Stage 3: build Chroma index (delegates to 03_chroma.py)."""
    _forward(ctx, stage3_chroma.main)


@app.command(
    "query",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def query_cmd(ctx: typer.Context) -> None:
    """Query the Chroma collection (delegates to query.py)."""
    _forward(ctx, query.main)


@app.command(
    "run",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def run_cmd(ctx: typer.Context) -> None:
    """Run the pipeline top-to-bottom (delegates to run_pipeline.py)."""
    _forward(ctx, run_pipeline.main)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
