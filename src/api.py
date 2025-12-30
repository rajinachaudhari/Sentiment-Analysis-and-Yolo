from fastapi import FastAPI
from pydantic import BaseModel

from .rag_engine import generate_answer
from .safety import is_unsafe, safe_answer

# Create FastAPI app
app = FastAPI()

# Request body structure
class QuestionRequest(BaseModel):
    question: str

# Health check (optional but good practice)
@app.get("/")
def home():
    return {"message": "RAG API is running"}

# Main endpoint
@app.post("/ask")
def ask_question(payload: QuestionRequest):

    question = payload.question

    # Safety check
    if is_unsafe(question):
        return {
            "answer": "Unsafe question detected.",
            "sources": [],
            "confidence": 0.0
        }

    # Generate answer using RAG
    answer = generate_answer(question)
    answer = safe_answer(answer)

    return {
        "answer": answer,
        "sources": [],
        "confidence": 0.8
    }
