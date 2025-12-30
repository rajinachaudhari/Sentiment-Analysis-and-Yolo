import pickle
from sentence_transformers import SentenceTransformer

# Load documents
with open("data/docs.txt") as f:
    documents = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(documents)

# Build FAISS index
import faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save vector store
with open("data/vector_store.pkl", "wb") as f:
    pickle.dump((index, documents), f)

print("✅ vector_store.pkl created successfully")
