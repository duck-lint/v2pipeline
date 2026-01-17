import chromadb
client = chromadb.PersistentClient(path=r"...\path\to\stage_3_chroma") 
c = client.get_collection("v1_chunks")
print(c.count())