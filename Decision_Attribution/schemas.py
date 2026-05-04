# schemas.py
from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str
    context: str
