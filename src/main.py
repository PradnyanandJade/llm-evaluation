from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.rag.pipeline import RAGPipeline
from src.api.chat_router import router as chat_router
from src.config import settings

import os

print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("API KEY EXISTS:", bool(os.getenv("LANGSMITH_API_KEY")))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App Starting...!")
    app.state.pipeline = RAGPipeline()
    yield
    app.state.pipeline = None
    print("App Terminated...!")

app = FastAPI(lifespan=lifespan)

app.include_router(chat_router,prefix="/api")



