from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

splitter = RecursiveCharacterTextSplitter(
    chunk_size = settings.CHUNK_SIZE,
    chunk_overlap = settings.CHUNK_OVERLAP
)