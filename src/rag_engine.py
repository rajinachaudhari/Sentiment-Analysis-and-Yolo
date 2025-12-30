import yaml
from transformers import pipeline
from .retrieval import retrieve_chunks

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

llm = pipeline(
    "text-generation",
    model=config["model"]["name"],
    max_new_tokens=config["model"]["max_tokens"]
)

def generate_answer(question):
    chunks = retrieve_chunks(question)

    if not chunks:
        return "No relevant data found."

    context = "\n".join(chunks)

    prompt = f"""
Use ONLY the information below to answer.

Context:
{context}

Question:
{question}

Answer:
"""

    return llm(prompt)[0]["generated_text"]
