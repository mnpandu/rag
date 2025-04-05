from load_data import load_and_split_text
from vector_store import create_vectorstore
from qa_model import load_qa_pipeline
from rag_runner import run_rag_questions

text = """
LangChain is a framework for developing applications powered by language models.
It provides tools to connect language models to sources of data and allow reasoning.
RAG is a pattern where a user query is augmented by retrieved documents from a vector store.
"""

# Step 1: Load and split text
texts = load_and_split_text(text)

# Step 2: Build retriever
retriever = create_vectorstore(texts)

# Step 3: Load QA pipeline
qa_pipeline = load_qa_pipeline()

# Step 4: Define and run questions
questions = [
    "What is LangChain?",
    "What does RAG stand for?",
    "How does RAG work?",
    "What are the main features of LangChain?"
]

run_rag_questions(questions, retriever, qa_pipeline)