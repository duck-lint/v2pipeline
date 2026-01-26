# RAG Pipeline v2

A local-first, reproducible Markdown -> Chroma ingestion pipeline designed for iterative experimentation with chunking strategies, metadata schemas, and embedding models.

The pipeline is explicitly staged to preserve provenance and enable clean, repeatable rebuilds as documents evolve.

Stages:
1. Snapshot raw inputs
2. Clean + normalize text
3. Chunk into JSONL with rich metadata
4. Embed and index into Chroma

---

## Design principles

- Local-first and reproducible: Stage 0 snapshots inputs so downstream stages operate on a stable, auditable corpus rather than a live vault.
- Clear stage boundaries: each stage emits concrete artifacts on disk, making failures inspectable and reruns deterministic.
- Typed CLI semantics: `*_path` denotes a file or file-or-folder input; `*_dir` always denotes a directory.
- Stable document identity: document IDs are stable across edits; chunk identities are deterministic within a document.
- Iteration-friendly indexing: Stage 3 supports rebuild, upsert, and delete-sync so the index stays consistent as notes evolve.

---

## Environment (Python 3.11)

Use a Python 3.11 virtual environment for this repo.
Stage 3 (embedding) supports CUDA only when running inside a venv that has a CUDA-enabled torch build.

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

If `cuda_available` is false:

- Confirm the venv is active: `where python` should point to `.venv`
- Check the torch build:
  ```bash
  python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
  ```
  (`torch.version.cuda` should not be `None`)
- Verify the GPU is visible to the OS/driver (`nvidia-smi`)
- Reinstall torch using a CUDA build compatible with your driver

---

## Quick start

### 1) Initialize stage folders

Creates the standard stage directories relative to the current working directory.

```bash
python init_folders.py --root .
```

---

### 2) Stage 0 -- snapshot raw inputs

Copies notes into `stage_0_raw` using `shutil.copy2()` to preserve filesystem timestamps.

This stage exists to:
- Freeze inputs for a given run (provenance)
- Avoid ingesting a moving target (live vault)
- Make downstream stages reproducible and auditable

```bash
python 00_copy_raw.py --input_path /path/to/vault --stage0_dir stage_0_raw
```

---

### 3) Stage 1 -- clean text

Removes noise (frontmatter handling, normalization) while preserving Obsidian wikilinks.

```bash
python 01_clean.py --stage0_path stage_0_raw --stage1_dir stage_1_clean
```

---

### 4) Stage 2 -- chunk into JSONL

Splits notes into paragraph-aligned chunks with stable document identity and structured metadata. When available, prefers Stage 1 output.

```bash
python 02_chunk.py \
  --stage0_path stage_0_raw \
  --stage1_dir stage_1_clean \
  --out_dir stage_2_chunks \
  --prefer_stage1
```

---

### 5) Merge per-file chunks

Combines per-note `*.chunks.jsonl` files into a single JSONL for indexing.

```bash
python merge_chunks_jsonl.py \
  --chunks_dir stage_2_chunks \
  --output_jsonl stage_2_chunks_merged.jsonl
```

---

### 6) Stage 3 -- build Chroma index

Embeds chunks and writes them into a persistent Chroma collection.

```bash
python 03_chroma.py \
  --chunks_jsonl stage_2_chunks_merged.jsonl \
  --persist_dir stage_3_chroma \
  --collection v1_chunks \
  --mode upsert
```

---

## Stage 3 modes

Rebuild (delete and recreate the collection):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode rebuild
```

Append (add only; fails if any IDs already exist):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode append
```

Upsert (default; overwrite existing IDs when supported):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode upsert
```

Sync deletes (remove stale chunk IDs when documents change):
```bash
python 03_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode upsert --sync_deletes
```

---

## Common flags

File discovery (Stages 0-2):
- `--no_recursive` to disable recursion when input is a folder
- `--exclude` (repeatable) to skip glob patterns (e.g. `.obsidian/**`)

YAML handling (Stages 1-2):
- `--yaml_mode strict|lenient`
  - `lenient` logs `yaml_error` and continues

Chunking behavior (Stage 2):
- `--prefer_stage1` uses cleaned text when available

---

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

---

## Notes

- `stage_2_chunks` contains per-note `*.chunks.jsonl` files; `merge_chunks_jsonl.py` only merges these artifacts.
- `stage_3_chroma` is a persistent Chroma directory. Use a new `--persist_dir` for test runs if you encounter file locks on Windows.
- Naming conventions:
  - `*_path` -> file path or file-or-folder input
  - `*_dir` -> directory (output or persistent storage)
