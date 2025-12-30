import yaml
import pickle
from sentence_transformers import SentenceTransformer

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(question):
    with open("data/vector_store.pkl", "rb") as f:
        index, documents = pickle.load(f)

    k = config["retrieval"]["top_k"]
    query_embedding = model.encode([question])
    _, indices = index.search(query_embedding, k)

    return [documents[i] for i in indices[0]]
