from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

def load_qa_pipeline(model_name="distilbert-base-cased-distilled-squad"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return qa_pipeline