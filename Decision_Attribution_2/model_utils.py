from transformers import pipeline

_summarizer = None
_qa_pipeline = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("[model_utils] Loading summarizer (BART Large)...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
    return _summarizer

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        print("[model_utils] Loading QA model (RoBERTa)...")
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
    return _qa_pipeline

def summarize_text(text: str) -> str:
    try:
        summarizer = get_summarizer()
        result = summarizer(
            text[:1024],
            max_length=130,
            min_length=30,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception as e:
        return f"[Summarization error: {e}]"

def answer_question(question: str, context: str):
    try:
        qa = get_qa_pipeline()
        result = qa(question=question, context=context[:1024])
        return result["answer"], round(result["score"], 4)
    except Exception as e:
        return f"[QA error: {e}]", 0.0