from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import torch
from pipeline.common import configure_stdout

_WARNED: set[str] = set()

def _warn_deprecated(flag: str, replacement: str) -> None:
    if flag in _WARNED:
        return
    _WARNED.add(flag)
    print(f"[deprecation] {flag} is deprecated; use {replacement}")


def preview(text: str, n: int = 180) -> str:
    t = " ".join(text.split())  # collapse whitespace for display only
    return (t[:n] + "â€¦") if len(t) > n else t


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", type=str, default="stage_3_chroma", help="default=stage_3_chroma")
    ap.add_argument("--persist_path", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--collection", type=str, default="v1_chunks", help="default=v1_chunks")
    ap.add_argument("--query", type=str, required=True, help="required=True")
    ap.add_argument("--k", type=int, default=5, help="default=5")
    ap.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="default=sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda | default=auto")
    ap.add_argument("--anchor", type=str, default="", help="Optional filter: chunk_anchor equals this value")
    ap.add_argument("--doc_id", type=str, default="")
    ap.add_argument("--doc_type", type=str, default="")
    ap.add_argument("--folder", type=str, default="")
    ap.add_argument("--sensitivity", type=str, default="")
    ap.add_argument("--rel_path_prefix", type=str, default="")
    ap.add_argument("--fetch_k", type=int, default=0)
    ap.add_argument("--show_meta", type=str, choices=["true", "false"], default="true")
    return ap


def main(argv: list[str] | None = None) -> int:
    configure_stdout(errors="replace")
    ap = build_parser()
    args = ap.parse_args(argv)

    print(f"[stage_04_query] args: {args}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    persist_dir = args.persist_dir
    if args.persist_path:
        _warn_deprecated("--persist_path", "--persist_dir")
        if args.persist_dir != "stage_3_chroma":
            raise ValueError("Use only one of --persist_dir or --persist_path")
        persist_dir = args.persist_path
    args.persist_dir = persist_dir

    persist_dir_path = Path(persist_dir).resolve()
    if not persist_dir_path.exists():
        raise FileNotFoundError(f"Missing persist_dir: {persist_dir_path}")
    if not persist_dir_path.is_dir():
        raise NotADirectoryError(f"persist_dir must be a directory: {persist_dir_path}")

    client = chromadb.PersistentClient(path=str(persist_dir_path))
    collection = client.get_collection(name=args.collection)

    model = SentenceTransformer(args.embed_model, device=device)
    q_emb = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    where = {}
    if args.anchor:
        where["chunk_anchor"] = args.anchor
    if args.doc_id:
        where["doc_id"] = args.doc_id
    if args.doc_type:
        where["doc_type"] = args.doc_type
    if args.folder:
        where["folder"] = args.folder
    if args.sensitivity:
        where["sensitivity"] = args.sensitivity
    if not where:
        where = None

    if args.rel_path_prefix:
        n_results_query = args.fetch_k if args.fetch_k > 0 else max(50, args.k * 5)
    else:
        n_results_query = args.k

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results_query,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if args.rel_path_prefix:
        prefix = args.rel_path_prefix.replace("\\", "/")
        filtered = []
        for doc, meta, dist in zip(docs, metas, dists):
            rel_path = meta.get("rel_path") if isinstance(meta, dict) else None
            rel_path = rel_path or ""
            rel_norm = rel_path.replace("\\", "/")
            if rel_norm.startswith(prefix):
                filtered.append((doc, meta, dist))
        docs, metas, dists = zip(*filtered) if filtered else ([], [], [])
        if len(docs) < args.k:
            print(
                f"[stage_04_query] warning: rel_path_prefix reduced results: "
                f"{len(docs)}/{args.k} (fetched {n_results_query})"
            )
        docs = list(docs[: args.k])
        metas = list(metas[: args.k])
        dists = list(dists[: args.k])
    else:
        docs = docs[: args.k]
        metas = metas[: args.k]
        dists = dists[: args.k]

    print(f"\nQuery: {args.query}")
    if where:
        print(f"Filter: {where}")
    print(f"Top k: {args.k}\n")

    show_meta = args.show_meta.lower() == "true"
    if show_meta and metas:
        print("Top-1 metadata:")
        print(json.dumps(metas[0], indent=2, ensure_ascii=False))
        print("")

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # Chroma distance depends on space; with normalized embeddings it's typically cosine distance-ish.
        rel_path = meta.get("rel_path")
        doc_type = meta.get("doc_type")
        folder = meta.get("folder")
        sensitivity = meta.get("sensitivity")
        heading = meta.get("heading_path_str") or meta.get("chunk_anchor")
        entry_date = meta.get("entry_date")
        source_date = meta.get("source_date")
        chunk_id = meta.get("chunk_id")

        print(
            f"[{rank}] dist={dist:.4f} | rel_path={rel_path} | doc_type={doc_type} | "
            f"folder={folder} | sensitivity={sensitivity} | entry_date={entry_date} | "
            f"source_date={source_date}"
        )
        if heading:
            print(f"     heading={heading}")
        print(f"     chunk_id={chunk_id}")
        print(f"     preview: {preview(doc)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
