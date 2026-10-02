from src.rag.retrieval.dense_retriever import DenseRetriever
from sentence_transformers import CrossEncoder
from src.config import settings
from langsmith import traceable

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    # retriever = DenseRetriever()
    # retriever = SparseRetriever()
    # retriever = HybridRetriever()
    def __init__(self):
        self.retriever=DenseRetriever()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.top_k = settings.TOP_K

    @traceable(name="Rerank")
    def rerank(self,query:str,documents):
        pairs = [(query,document.page_content) for document in documents]
        scores = self.reranker.predict(pairs)
        reranked_documents = sorted(
            zip(documents,scores),
            key=lambda x:x[1],
            reverse=True
        )[:self.top_k]
        return reranked_documents
    
    @traceable(name="Reranker Retriever")
    def retrieve(self,query:str):
        documents = self.retriever.retrieve(query=query)
        reranked_documents = self.rerank(
            query=query,
            documents=documents
        )
        documents = [document for document,score in reranked_documents]
        return documents
