from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.sparse_retriever import SparseRetriever
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.reranker import Reranker
from src.config import settings
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
from deepeval.evaluate import evaluate,CacheConfig
from evals.helper import load_goldens,get_metrics


# WE SELECTED DENSE + RERANKING WAY AS IT WAS EFFICIENT IN ALL OTHER WAYS 
# DENSE + RERANKING ====> Contextual Recall -> 1.00  , Contextual Precision -> 0.92   

GOLDEN_PATH = "./goldens/retriever_goldens.json"
JUDGE_MODEL = settings.OPENAI_MODEL_NAME
THRESHOLD = 0.7

# retriever = DenseRetriever()  # no reranking
# retriever = SparseRetriever() # no reranking
# retriever = HybridRetriever() # no reranking

retriever = Reranker() # Internally uses Dense Only Retriever

def run():
    goldens = load_goldens(GOLDEN_PATH)
    test_cases = []
    for g in goldens:
        retrieved = retriever.retrieve(query=g["question"])
        retriever_context = [doc.page_content for doc in retrieved]
        test_cases.append( 
            LLMTestCase(
                input=g['question'],
                expected_output=g['ideal_answer'],
                retrieval_context=retriever_context
            )
        )
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True)
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker_with_dense_retriever",
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL_NAME,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
        cache_config=CacheConfig(
            use_cache=False,
            write_cache=False
        )
    )
    return get_metrics(result=result)

if __name__=="__main__":
    run()

