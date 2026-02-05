import faiss
import pickle
from sentence_transformers import SentenceTransformer


# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DIM = 384  # embedding size for MiniLM
index = faiss.IndexFlatIP(DIM)  # cosine similarity
metadata = []  # stores original messages

def add_messages(messages: list[str]):
    embeddings = embedder.encode(messages, normalize_embeddings=True)
    index.add(embeddings)
    metadata.extend(messages)
    print(index)

def search_similar(text: str, k=3):
    embedding = embedder.encode([text], normalize_embeddings=True)
    scores, indices = index.search(embedding, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx != -1:
            results.append({
                "text": metadata[idx],
                "similarity": float(score)
            })
    return results

def save_to_disk():
    faiss.write_index(index, "scam_vectors.index")
    with open("scam_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def on_restart():
    index = faiss.read_index("scam_vectors.index")
    with open("scam_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)


