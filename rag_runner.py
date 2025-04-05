def run_rag_questions(questions, retriever, qa_pipeline):
    for question in questions:
        print(f"Question: {question}")
        try:
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            result = qa_pipeline(question=question, context=context)
            print(f"Answer: {result['answer']}\n")
            print("Source documents:")
            for doc in docs:
                print(f"- {doc.page_content.strip()}\n")
        except Exception as e:
            print(f"Error: {e}")
        print("=" * 50 + "\n")