# RAG Pipeline v2

Markdown-to-Chroma pipeline with four stages:
1) copy raw notes, 2) clean, 3) chunk to JSONL, 4) embed + index.

## Environment (Python 3.11)

Use a Python 3.11 virtual environment for this repo. Stage 3 CUDA only works if you run
inside the venv that has the CUDA-enabled torch build.

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

If `cuda_available` is false:
- Confirm you're in the venv: `where python` should point to `.venv`.
- Check the torch build: `python -c "import torch; print(torch.__version__); print(torch.version.cuda)"` (the CUDA field should not be `None`).
- Verify the GPU is visible to the OS/driver (`nvidia-smi` should list your grahics card).
- Reinstall torch with a supported CUDA build for your driver (use the selector link above).

## Quick start

1) Initialize stage folders
```bash
python init_folders.py --root .
```

2) Stage 0: copy raw notes
```bash
python 00_copy_raw.py --input_path /path/to/vault --stage0_dir stage_0_raw
```

3) Stage 1: clean (preserves wikilinks)
```bash
python 01_clean.py --stage0_path stage_0_raw --stage1_dir stage_1_clean
```

4) Stage 2: chunk (prefer Stage 1 output)
```bash
python 02_chunk.py --stage0_path stage_0_raw --stage1_dir stage_1_clean --out_dir stage_2_chunks --prefer_stage1
```

5) Merge per-file JSONL into one
```bash
python merge_chunks_jsonl.py --chunks_dir stage_2_chunks --output_jsonl stage_2_chunks_merged.jsonl
```

6) Stage 3: build Chroma index
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --persist_dir stage_3_chroma --collection v1_chunks --mode upsert
```

## Stage 3 modes

Rebuild (delete and recreate the collection):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode rebuild
```

Append (add only; fails if any ids already exist):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode append
```

Upsert (default; overwrite existing ids when supported):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode upsert
```

Sync deletes (remove stale chunk_ids when chunks change):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode upsert --sync_deletes
```

## Common flags

Stage 0/1/2 file discovery:
- `--no_recursive` to disable recursion
- `--exclude` (repeatable) to skip globs, e.g. `.obsidian/**`

Stage 1/2 YAML behavior:
- `--yaml_mode strict|lenient` (lenient logs `yaml_error` and continues)

Stage 2 chunking:
- `--prefer_stage1` uses `stage_1_clean` when present

## Querying

Basic query:
```bash
python query.py --persist_dir stage_3_chroma --collection v1_chunks --query "assumption ledger"
```

Exact-match filters:
```bash
python query.py --persist_dir stage_3_chroma --collection v1_chunks --query "assumption ledger" --doc_type inbox --folder INBOX
```

Prefix filter (post-filtered):
```bash
python query.py --persist_dir stage_3_chroma --collection v1_chunks --query "assumption ledger" --rel_path_prefix "INBOX/"
```

## Notes

- `stage_2_chunks` contains per-file `*.chunks.jsonl`. `merge_chunks_jsonl.py` only merges those files.
- `stage_3_chroma` is a Chroma persistent directory. Use a new `--persist_dir` for test runs if you hit file locks on Windows.
- Naming convention:
    - `*_path` is a file path or file-or-folder input.
    - `*_dir` is always a directory.
