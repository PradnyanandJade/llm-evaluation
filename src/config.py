from pydantic_settings import BaseSettings,SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    HF_TOKEN : str
    OPENAI_API_KEY : str
    OPENAI_MODEL_NAME : str
    OPENAI_EMBEDDING_MODEL_NAME : str
    CHUNK_SIZE : int
    CHUNK_OVERLAP : int
    TOP_K_RETRIEVER : int
    TOP_K_HYBRID_SEARCH : int
    TOP_K : int
    LANGSMITH_TRACING: bool 
    LANGSMITH_ENDPOINT: str 
    LANGSMITH_API_KEY: str 
    LANGSMITH_PROJECT: str 
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()