from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.sparse_retriever import SparseRetriever
from src.config import settings
from langsmith import traceable

class HybridRetriever:
    
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.top_k_hybrid_search = settings.TOP_K_HYBRID_SEARCH

    @traceable(name="Reciprocal Rank Fusion")
    def reciprocal_rank_fusion(self,results_list,k=60):
        scores = {}
        documents = {}
        for results in results_list:
            for rank,document in enumerate(results,start=1):
                doc_id = document.id
                score = 1 / (k + rank)
                scores[doc_id] = scores.get(doc_id,0) + score
                documents[doc_id] = document

        ranked_documents = sorted(
            documents.values(),
            key= lambda document:scores[document.id],
            reverse=True
        )
        return ranked_documents 
        
    
    @traceable(name="Hybrid Retriever")
    def retrieve(self,query:str):
        dense_retrieval_documents = self.dense_retriever.retrieve(query=query)
        sparse_retrieval_documents = self.sparse_retriever.retrieve(query=query)
        results_list = [dense_retrieval_documents,sparse_retrieval_documents]
        fused_documents = self.reciprocal_rank_fusion(
            results_list=results_list
        )
        return fused_documents[:self.top_k_hybrid_search]