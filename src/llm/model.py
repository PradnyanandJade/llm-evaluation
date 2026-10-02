from langchain_openai import ChatOpenAI
from src.config import settings

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    max_completion_tokens=200,
    temperature=0.1,
    api_key=settings.OPENAI_API_KEY
)