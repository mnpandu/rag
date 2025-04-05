# 🤖 Retrieval-Augmented Generation (RAG) with HuggingFace and LangChain

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using:

- 🧠 **HuggingFace QA Models** (e.g., `distilbert-base-cased-distilled-squad`)
- 🔍 **LangChain Vector Search** (FAISS + HuggingFace embeddings)
- 📄 **Your own custom text data**
- 🧱 Fully modular Python design

## Features

- Vector-based document retrieval with FAISS
- Contextual question answering with HuggingFace
- RAG pattern implemented in a modular and maintainable way
- Works with CPU and GPU

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
rag_pipeline/
├── main.py
├── load_data.py
├── vector_store.py
├── qa_model.py
├── rag_runner.py
├── requirements.txt
└── README.md
```

## How It Works

### RAG Flow

```
[Question]
    ↓
[Retriever] ← FAISS + Embeddings
    ↓
[Context from Top-k Docs]
    ↓
[HuggingFace QA Model]
    ↓
[Answer]
```

## Running the App

```bash
python main.py
```

## Example Questions

- What is LangChain?
- What does RAG stand for?
- How does RAG work?
- What are the main features of LangChain?

## Notes

- Replace the input `text` with your own documents or load from PDFs, CSVs, etc.
- Easily extendable to Chainlit, LangGraph, or FastAPI interfaces.

## License

MIT License – feel free to use, fork, and modify!