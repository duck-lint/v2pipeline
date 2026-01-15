from __future__ import annotations

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import torch

PIPELINE_VERSION = "v1"
STAGE3_VERSION = "v0.1"

CHROMA_META_KEYS = [
    "doc_id",
    "chunk_id",
    "chunk_anchor",
    "chunk_title",
    "chunk_index",
    "rel_path",
    "entry_date",
    "source_date",
    "source_hash",
    "content_hash",
    "embed_model",
    "embed_dim",
    "chunker_version",
]

def stable_settings_hash(d: Dict[str, Any]) -> str:
    # Stable hash of settings (sorted keys) so runs are comparable
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batch(iterable: List[Any], n: int) -> List[List[Any]]:
    return [iterable[i : i + n] for i in range(0, len(iterable), n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="stage_2_chunks.jsonl", help="default=stage_2_chunks.jsonl")
    ap.add_argument("--db_dir", type=str, default="stage_3_chroma", help="default=stage_3_chroma")
    ap.add_argument("--collection", type=str, default="v1_chunks", help="default=v1_chunks")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda | default=auto")
    ap.add_argument("--batch_size", type=int, default=32, help="default=32")
    ap.add_argument("--reset_db", action="store_true", help="Delete stage_3_chroma/ before building")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"[stage_03_chroma] args: {args}")

    chunks_path = Path(args.chunks_jsonl).resolve()
    db_dir = Path(args.db_dir).resolve()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    # Decide device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    settings = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "chunks_jsonl": str(chunks_path),
        "db_dir": str(db_dir),
        "collection": args.collection,
        "embed_model": args.embed_model,
        "device": device,
        "batch_size": args.batch_size,
        "reset_db": args.reset_db,
    }
    settings_hash = stable_settings_hash(settings)

    rows = load_jsonl(chunks_path)
    ids = [r["metadata"]["chunk_id"] for r in rows]
    docs = [r["text"] for r in rows]
    metas = []
    for r in rows:
        m = r["metadata"]
        metas.append({k: m.get(k) for k in CHROMA_META_KEYS if m.get(k) is not None})

    # Basic sanity
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id detected in stage_2_chunks.jsonl")

    print("[stage_3] ---- input summary ----")
    print(f"[stage_3] chunks_jsonl={chunks_path}")
    print(f"[stage_3] rows={len(rows)}")
    print(f"[stage_3] embed_model={args.embed_model}")
    print(f"[stage_3] device={device} | cuda_available={torch.cuda.is_available()}")
    print(f"[stage_3] settings_hash={settings_hash}")

    if args.dry_run:
        print("[stage_3] dry_run=True (not embedding / not writing DB)")
        return

    # Reset DB directory if requested
    if args.reset_db and db_dir.exists():
        shutil.rmtree(db_dir)
        print(f"[stage_3] deleted db_dir: {db_dir}")

    db_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = SentenceTransformer(args.embed_model, device=device)

    # Create Chroma persistent client
    client = chromadb.PersistentClient(path=str(db_dir))

    # Recreate collection (V1: overwrite)
    try:
        client.delete_collection(name=args.collection)
        print(f"[stage_3] deleted existing collection: {args.collection}")
    except Exception:
        pass

    collection = client.create_collection(
        name=args.collection,
        metadata={
            "pipeline_version": PIPELINE_VERSION,
            "stage3_version": STAGE3_VERSION,
            "embed_model": args.embed_model,
            "device": device,
            "settings_hash": settings_hash,
        },
    )

    # Embed and add in batches
    total = 0
    for idxs in batch(list(range(len(docs))), args.batch_size):
        batch_docs = [docs[i] for i in idxs]
        batch_ids = [ids[i] for i in idxs]
        batch_metas = [metas[i] for i in idxs]

        embeddings = model.encode(
            batch_docs,
            batch_size=min(args.batch_size, len(batch_docs)),
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-friendly
            show_progress_bar=False,
        )

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings.tolist(),
        )
        total += len(batch_docs)

    print("[stage_3] ---- build summary ----")
    print(f"[stage_3] wrote collection={args.collection} | total_added={total}")
    print(f"[stage_3] db_dir={db_dir}")

    # Write run manifest
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "settings": settings,
        "settings_hash": settings_hash,
        "counts": {"chunks": len(rows), "added": total},
    }
    manifest_path = Path("run_manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage_3] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
