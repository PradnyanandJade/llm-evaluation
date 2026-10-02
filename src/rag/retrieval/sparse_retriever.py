from langchain_community.retrievers import BM25Retriever
from src.config import settings
from src.rag.ingestion.ingest import get_documents
from langchain_core.documents import Document
from langsmith import traceable

class SparseRetriever:

    def __init__(self):
        original_documents = get_documents()
        normalized_documents = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in original_documents
        ]
        self.retriever = BM25Retriever.from_documents(normalized_documents)
        self.retriever.k = settings.TOP_K_RETRIEVER 

    @traceable(name="Sparse Retriever")
    def retrieve(self,query:str):
        query = query.lower()
        return self.retriever.invoke(query) 

    