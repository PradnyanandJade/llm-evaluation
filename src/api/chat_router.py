from src.api.chat_request import ChatRequest
from fastapi import APIRouter,Request
from langsmith import traceable

router = APIRouter()

@router.post("/chat")
@traceable(name="Chat API")
def chat(request : Request,chat_request: ChatRequest):
    return request.app.state.pipeline.run(
        question=chat_request.question
    )