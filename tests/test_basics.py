from src.retrieval import retrieve_chunks
from src.rag_engine import generate_answer

def test_retrieval():
    result = retrieve_chunks("refund policy")
    assert len(result) > 0
    print("Retrieval Test Passed")

def test_generation():
    answer = generate_answer("What is the refund policy?")
    assert len(answer) > 10
    print("Generation Test Passed")

if __name__ == "__main__":
    test_retrieval()
    test_generation()
