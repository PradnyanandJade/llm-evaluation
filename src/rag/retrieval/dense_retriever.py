from src.rag.vector_store.chroma import vector_store
from src.config import settings
from langsmith import traceable

class DenseRetriever:
    
    def __init__(self):
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":settings.TOP_K_RETRIEVER,
                "fetch_k":20,
                "lambda_mult":0.5
            }
        )
    @traceable(name="Dense Retriever")
    def retrieve(self, query: str):
        return self.retriever.invoke(query)