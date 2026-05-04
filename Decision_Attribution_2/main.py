from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from schemas import SummarizeRequest, QARequest
from model_utils import summarize_text, answer_question
from explainer import (
    get_lime_explanation,
    get_shap_explanation,
    get_sentence_attribution
)

app = FastAPI(title="Decision Attribution In LLM's")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ✅ Summarization API
@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    summary = summarize_text(req.text)

    # 🔥 Attribution on INPUT (correct for your project)
    lime_exp = get_lime_explanation(req.text)
    shap_exp = get_shap_explanation(req.text)
    sentence_attr = get_sentence_attribution(req.text)

    return {
        "summary": summary,
        "lime": lime_exp,
        "shap": shap_exp,
        "sentence_attribution": sentence_attr,
    }


# ✅ QnA API
@app.post("/api/qa")
async def qa(req: QARequest):
    answer, score = answer_question(req.question, req.context)

    # 🔥 Attribution on INPUT CONTEXT (correct)
    lime_exp = get_lime_explanation(req.context)
    shap_exp = get_shap_explanation(req.context)
    sentence_attr = get_sentence_attribution(req.context)

    return {
        "answer": answer,
        "confidence": score,
        "lime": lime_exp,
        "shap": shap_exp,
        "sentence_attribution": sentence_attr,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}