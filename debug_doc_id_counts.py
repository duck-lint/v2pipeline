import chromadb
from pathlib import Path

PERSIST_DIR = r"path\to\dir"
COLLECTION = "v1_chunks"
DOC_ID = "doc_id"

client = chromadb.PersistentClient(path=str(Path(PERSIST_DIR)))
coll = client.get_collection(name=COLLECTION)

try:
    res = coll.get(where={"doc_id": DOC_ID}, include=[])
except TypeError:
    res = coll.get(where={"doc_id": DOC_ID})
ids = res.get("ids", [])
print("doc_id:", DOC_ID)
print("chunk_count:", len(ids))
print("sample_ids:", ids[:10])
