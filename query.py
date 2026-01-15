from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import torch


def preview(text: str, n: int = 180) -> str:
    t = " ".join(text.split())  # collapse whitespace for display only
    return (t[:n] + "…") if len(t) > n else t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", type=str, default="stage_3_chroma", help="default=stage_3_chroma")
    ap.add_argument("--collection", type=str, default="v1_chunks", help="default=v1_chunks")
    ap.add_argument("--query", type=str, required=True, help="required=True")
    ap.add_argument("--k", type=int, default=5, help="default=5")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda | default=auto")
    ap.add_argument("--anchor", type=str, default="", help="Optional filter: chunk_anchor equals this value")
    args = ap.parse_args()

    print(f"[stage_04_query] args: {args}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    db_dir = Path(args.db_dir).resolve()
    if not db_dir.exists():
        raise FileNotFoundError(f"Missing db_dir: {db_dir}")

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(name=args.collection)

    model = SentenceTransformer(args.embed_model, device=device)
    q_emb = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    where = {"chunk_anchor": args.anchor} if args.anchor else None

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=args.k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"\nQuery: {args.query}")
    if where:
        print(f"Filter: {where}")
    print(f"Top k: {args.k}\n")

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # Chroma distance depends on space; with normalized embeddings it's typically cosine distance-ish.
        rel_path = meta.get("rel_path")
        anchor = meta.get("chunk_anchor")
        title = meta.get("chunk_title")  # if you added it
        entry_date = meta.get("entry_date")
        source_date = meta.get("source_date")
        chunk_id = meta.get("chunk_id")

        print(f"[{rank}] dist={dist:.4f} | rel_path={rel_path} | anchor={anchor} | entry_date={entry_date} | source_date={source_date}")
        if title:
            print(f"     title={title}")
        print(f"     chunk_id={chunk_id}")
        print(f"     preview: {preview(doc)}\n")


if __name__ == "__main__":
    main()
