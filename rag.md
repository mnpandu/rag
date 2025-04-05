# Retrieval-Augmented Generation (RAG) Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, HuggingFace models, and FAISS vector store. The system retrieves relevant documents and generates answers to questions based on the retrieved context.

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [When to Use RAG](#when-to-use-rag)
3. [Why Use RAG?](#why-use-rag)
4. [How This Implementation Works](#how-this-implementation-works)
5. [Step-by-Step Process](#step-by-step-process)
6. [Flow Diagram](#flow-diagram)
7. [Example Usage](#example-usage)
8. [Requirements](#requirements)
9. [Installation](#installation)
10. [Customization](#customization)

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that:
- Combines information retrieval with text generation
- Augments language model knowledge with external data
- Retrieves relevant documents before generating answers
- Provides source attribution for generated answers

## When to Use RAG
- When you need answers based on specific documents
- When you want to limit responses to a knowledge base
- When source attribution is important
- When dealing with frequently updated information

## Why Use RAG?
- Overcomes fixed knowledge limitations of LLMs
- Reduces hallucination by grounding answers in documents
- More cost-effective than fine-tuning for new information
- Allows dynamic updates to the knowledge base

## How This Implementation Works
1. Text is split into manageable chunks
2. Chunks are converted to vector embeddings
3. Embeddings are stored in FAISS for efficient search
4. User questions are converted to embeddings
5. Relevant document chunks are retrieved
6. A QA model generates answers from retrieved context

## Step-by-Step Process

```python
# 1. Split text into chunks
text_splitter = CharacterTextSplitter(...)
texts = [t for t in text_splitter.split_text(text) if t.strip()]

# 2. Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(...)
db = FAISS.from_texts(texts, embeddings)

# 3. Load QA model
model_name = "distilbert-base-cased-distilled-squad"
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 4. Process questions
retriever = db.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(question)
context = "\n".join([doc.page_content for doc in docs])
answer = qa_pipeline(question=question, context=context)
```

## Flow Diagram

```text
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Input Text  │ →  │ Text Split  │ →  │ Create Embeddings│ →  │ Vector Store│
└─────────────┘    └─────────────┘    └──────────────────┘    └─────────────┘
                                                                     ↑
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐           │
│  Question   │ →  │  Embed      │ →  │ Retrieve Relevant│ ──────────┘
└─────────────┘    └─────────────┘    └──────────────────┘    
                                                             ┌─────────────┐
                                                             │ Generate    │
                                                             │ Answer      │
                                                             └─────────────┘
```

## Example Usage
questions = [
    "What is LangChain?",
    "What does RAG stand for?",
    "How does RAG work?"
]

for question in questions:
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    answer = qa_pipeline(question=question, context=context)
    
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print("Sources:")
    for doc in docs:
        print(f"- {doc.page_content}")                                                            

## Sample output
Question: What is LangChain?
Answer: LangChain is a framework for developing applications powered by language models.
Sources:
- LangChain is a framework for developing applications powered by language models.

## Installation
pip install torch transformers langchain langchain-community faiss-cpu sentence-transformers