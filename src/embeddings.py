import yaml
from sentence_transformers import SentenceTransformer

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
