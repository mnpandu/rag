from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def create_vectorstore(texts: list[str]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    db = FAISS.from_texts(texts, embeddings)
    return db.as_retriever(search_kwargs={"k": 2})