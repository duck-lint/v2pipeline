from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import torch
from common import configure_stdout

PIPELINE_VERSION = "v1"
STAGE3_VERSION = "v0.1"

CHROMA_META_KEYS = [
    "doc_id",
    "chunk_id",
    "chunk_key",
    "chunk_hash",
    "chunk_anchor",
    "chunk_title",
    "heading_path_str",
    "chunk_index",
    "rel_path",
    "source_uri",
    "cleaned_text",
    "entry_date",
    "source_date",
    "source_hash",
    "content_hash",
    "embed_model",
    "embed_dim",
    "chunker_version",
    "doc_type",
    "sensitivity",
    "folder",
]

def stable_settings_hash(d: Dict[str, Any]) -> str:
    # Stable hash of settings (sorted keys) so runs are comparable
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batch(iterable: List[Any], n: int) -> List[List[Any]]:
    return [iterable[i : i + n] for i in range(0, len(iterable), n)]

def find_existing_ids(collection: Any, ids: List[str], batch_size: int) -> List[str]:
    existing: List[str] = []
    for idxs in batch(list(range(len(ids))), batch_size):
        batch_ids = [ids[i] for i in idxs]
        try:
            res = collection.get(ids=batch_ids, include=[])
        except TypeError:
            res = collection.get(ids=batch_ids)
        existing.extend(res.get("ids", []))
    return existing

def get_existing_meta_by_id(collection: Any, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    try:
        res = collection.get(ids=ids, include=["metadatas"])
    except TypeError:
        res = collection.get(ids=ids)
    got_ids = res.get("ids", [])
    metas = res.get("metadatas", []) or []
    return {i: m for i, m in zip(got_ids, metas) if m is not None}


def main() -> None:
    configure_stdout(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="stage_2_chunks.jsonl", help="default=stage_2_chunks.jsonl")
    ap.add_argument("--persist_dir", type=str, default="stage_3_chroma", help="default=stage_3_chroma")
    ap.add_argument("--collection", type=str, default="v1_chunks", help="default=v1_chunks")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda | default=auto")
    ap.add_argument("--batch_size", type=int, default=32, help="default=32")
    ap.add_argument("--mode", type=str, choices=["rebuild", "append", "upsert"], default="upsert")
    ap.add_argument("--skip_unchanged", action="store_true", help="When upserting, skip chunks whose hash hasn't changed")
    ap.add_argument("--sync_deletes", action="store_true", help="Delete stale chunks not present in input (upsert only)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"[stage_03_chroma] args: {args}")

    chunks_path = Path(args.chunks_jsonl).resolve()
    persist_dir = Path(args.persist_dir).resolve()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    if not args.collection or not args.collection.strip():
        raise ValueError("--collection must be a non-empty name")
    if args.sync_deletes and args.mode != "upsert":
        raise ValueError("--sync_deletes is only valid with --mode upsert")

    # Decide device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    required_keys = ["chunk_id", "chunk_key", "chunk_hash", "source_uri", "chunk_index", "cleaned_text"]
    run_settings = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "chunks_jsonl": str(chunks_path),
        "persist_dir": str(persist_dir),
        "collection": args.collection,
        "embed_model": args.embed_model,
        "device": device,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "skip_unchanged": args.skip_unchanged,
        "sync_deletes": args.sync_deletes,
    }
    hash_settings = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "embed_model": args.embed_model,
        "normalize_embeddings": True,
        "chroma_meta_keys": CHROMA_META_KEYS,
        "required_keys": required_keys,
    }
    settings_hash = stable_settings_hash(hash_settings)

    print(
        f"[stage_3] start persist_dir={persist_dir} | collection={args.collection} | "
        f"mode={args.mode}"
    )
    print("[stage_3] ---- input summary ----")
    print(f"[stage_3] chunks_jsonl={chunks_path}")
    print(f"[stage_3] embed_model={args.embed_model}")
    print(f"[stage_3] device={device} | cuda_available={torch.cuda.is_available()}")
    print(f"[stage_3] settings_hash={settings_hash}")

    if args.dry_run:
        print("[stage_3] dry_run=True (not embedding / not writing DB)")
        return

    persist_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = SentenceTransformer(args.embed_model, device=device)

    # Create Chroma persistent client
    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.mode == "rebuild":
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
    else:
        try:
            collection = client.get_collection(name=args.collection)
        except Exception:
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
            print(f"[stage_3] created collection: {args.collection}")
        else:
            print(f"[stage_3] using existing collection: {args.collection}")
            coll_meta = getattr(collection, "metadata", None) or {}
            coll_model = coll_meta.get("embed_model")
            coll_hash = coll_meta.get("settings_hash")
            if coll_model and coll_model != args.embed_model:
                raise ValueError(
                    "Collection embed_model mismatch. "
                    "Use --mode rebuild or a new --collection name."
                )
            if coll_hash and coll_hash != settings_hash:
                raise ValueError(
                    "Collection settings_hash mismatch. "
                    "Use --mode rebuild or a new --collection name."
                )

    incoming_ids_by_doc: Dict[str, Set[str]] = {}
    first_seen: Dict[str, Tuple[int, str]] = {}
    rows_seen = 0
    embedded_or_upserted = 0
    skipped_unchanged = 0

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal embedded_or_upserted, skipped_unchanged, batch_ids, batch_docs, batch_metas
        if not batch_ids:
            return

        if args.mode == "append":
            existing = find_existing_ids(collection, batch_ids, batch_size=len(batch_ids))
            if existing:
                sample = existing[:10]
                raise ValueError(
                    f"Append mode found {len(existing)} duplicate ids. Sample: {sample}"
                )

        if args.mode == "upsert":
            if args.skip_unchanged:
                existing_meta = get_existing_meta_by_id(collection, batch_ids)
                keep_ids: List[str] = []
                keep_docs: List[str] = []
                keep_metas: List[Dict[str, Any]] = []
                for i, cid in enumerate(batch_ids):
                    prev = existing_meta.get(cid)
                    if prev and prev.get("chunk_hash") == batch_metas[i].get("chunk_hash"):
                        skipped_unchanged += 1
                        continue
                    keep_ids.append(cid)
                    keep_docs.append(batch_docs[i])
                    keep_metas.append(batch_metas[i])
                if not keep_ids:
                    batch_ids = []
                    batch_docs = []
                    batch_metas = []
                    return
                batch_ids = keep_ids
                batch_docs = keep_docs
                batch_metas = keep_metas

            embeddings = model.encode(
                batch_docs,
                batch_size=min(args.batch_size, len(batch_docs)),
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine-friendly
                show_progress_bar=False,
            )
            embeddings_list = embeddings.tolist()
            if hasattr(collection, "upsert"):
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings_list,
                )
            else:
                existing = find_existing_ids(collection, batch_ids, batch_size=len(batch_ids))
                if existing:
                    collection.delete(ids=existing)
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings_list,
                )
            embedded_or_upserted += len(batch_docs)
        else:
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
            embedded_or_upserted += len(batch_docs)

        batch_ids = []
        batch_docs = []
        batch_metas = []

    for r in iter_jsonl(chunks_path):
        rows_seen += 1
        m = r.get("metadata", {})
        missing = False
        for k in required_keys:
            if k not in m or m.get(k) in (None, ""):
                missing = True
                break
        heading_path = m.get("heading_path") or []
        if isinstance(heading_path, list):
            heading_path_str = " > ".join(heading_path)
        else:
            heading_path_str = str(heading_path)
        if heading_path_str is None:
            missing = True
        if missing:
            raise ValueError(f"Missing required chunk metadata at row {rows_seen}")

        chunk_id = m["chunk_id"]
        source_uri = str(m.get("source_uri") or "")
        if chunk_id in first_seen:
            first_row, first_uri = first_seen[chunk_id]
            raise ValueError(
                "Duplicate chunk_id detected: "
                f"{chunk_id} at row {rows_seen} (source_uri={source_uri}); "
                f"first seen at row {first_row} (source_uri={first_uri})"
            )
        first_seen[chunk_id] = (rows_seen, source_uri)

        doc_id = m.get("doc_id")
        if args.sync_deletes and not doc_id:
            raise ValueError(f"sync_deletes requires doc_id at row {rows_seen}")
        if doc_id:
            incoming_ids_by_doc.setdefault(str(doc_id), set()).add(chunk_id)

        base = {k: m.get(k) for k in CHROMA_META_KEYS if k != "heading_path_str" and m.get(k) is not None}
        base["heading_path_str"] = heading_path_str

        batch_ids.append(chunk_id)
        batch_docs.append(r["text"])
        batch_metas.append(base)

        if len(batch_ids) >= args.batch_size:
            flush_batch()

    flush_batch()

    docs_synced = 0
    chunks_deleted = 0
    if args.sync_deletes:
        for doc_id, incoming_ids in incoming_ids_by_doc.items():
            try:
                res = collection.get(where={"doc_id": doc_id}, include=[])
            except TypeError:
                res = collection.get(where={"doc_id": doc_id})
            existing_ids = set(res.get("ids", []))
            to_delete = list(existing_ids - incoming_ids)
            if not to_delete:
                docs_synced += 1
                continue
            for idxs in batch(list(range(len(to_delete))), 256):
                batch_del = [to_delete[i] for i in idxs]
                collection.delete(ids=batch_del)
                chunks_deleted += len(batch_del)
            docs_synced += 1

        print(f"[stage_3] sync_deletes docs_synced={docs_synced} | chunks_deleted={chunks_deleted}")

    print("[stage_3] ---- build summary ----")
    print(f"[stage_3] wrote collection={args.collection} | total_added={embedded_or_upserted}")
    print(f"[stage_3] persist_dir={persist_dir}")

    # Write run manifest
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "run_settings": run_settings,
        "hash_settings": hash_settings,
        "settings_hash": settings_hash,
        "counts": {
            "rows_seen": rows_seen,
            "embedded_or_upserted": embedded_or_upserted,
            "skipped_unchanged": skipped_unchanged,
            "docs_synced": docs_synced,
            "chunks_deleted": chunks_deleted,
        },
    }
    manifest_path = persist_dir / f"run_manifest_{ts}_{settings_hash}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage_3] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
