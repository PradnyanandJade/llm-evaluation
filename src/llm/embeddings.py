from langchain_openai import OpenAIEmbeddings
from src.config import settings

embedding = OpenAIEmbeddings(
    model=settings.OPENAI_EMBEDDING_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY
)

