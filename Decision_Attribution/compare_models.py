import time
from model_utils import summarize_text, answer_question

# 👉 Change this import depending on which project you're in
# For Variant 1: use original model_utils
# For Variant 2: use modified model_utils

def evaluate_summarization(text):
    start = time.time()
    summary = summarize_text(text)
    end = time.time()

    return {
        "summary": summary,
        "time": round(end - start, 4),
        "length": len(summary.split())
    }


def evaluate_qa(question, context):
    start = time.time()
    answer, score = answer_question(question, context)
    end = time.time()

    return {
        "answer": answer,
        "confidence": score,
        "time": round(end - start, 4),
        "answer_length": len(answer.split())
    }


# 🧪 Test Input
context = """
Cells contain organelles that perform specific functions. The nucleus controls activities, and ribosomes make proteins. 
Mitochondria is the powerhouse of the cell because it produces energy. Lysosomes break down waste.
"""

question = "What is the powerhouse of the cell?"

# 🔥 Run evaluation
print("\n--- SUMMARIZATION ---")
sum_result = evaluate_summarization(context)

print(f"Summary: {sum_result['summary']}")
print(f"Time Taken: {sum_result['time']} sec")
print(f"Summary Length: {sum_result['length']} words")


print("\n--- QUESTION ANSWERING ---")
qa_result = evaluate_qa(question, context)

print(f"Answer: {qa_result['answer']}")
print(f"Confidence: {qa_result['confidence']}")
print(f"Time Taken: {qa_result['time']} sec")
print(f"Answer Length: {qa_result['answer_length']} words")