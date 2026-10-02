from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric,ContextualRelevancyMetric,FaithfulnessMetric,AnswerRelevancyMetric
from deepeval.evaluate import evaluate,CacheConfig
from src.rag.pipeline import RAGPipeline
from src.config import settings
from evals.helper import load_goldens,get_metrics


GOLDEN_PATH = "./goldens/pipeline_goldens.json"
JUDGE_MODEL = settings.OPENAI_MODEL_NAME
THRESHOLD = 0.7

pipeline = RAGPipeline()

def document_to_text(documents):
    return [document.page_content for document in documents]

def run():
    goldens = load_goldens(GOLDEN_PATH)
    test_cases = []
    for g in goldens:
        question = g["question"]
        expected_output = g["expected_output"]
        result = pipeline.run(question=question)
        answer = result["answer"]
        documents = result["documents"]
        test_cases.append(
            LLMTestCase(
                input=question,
                expected_output=expected_output,
                actual_output=answer,
                retrieval_context=document_to_text(documents=documents)
            )
        )
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        ContextualRelevancyMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True)
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "pipeline": "RAGPipeline",
            "retriever": "dense + reranker",
            "generator": settings.OPENAI_MODEL_NAME,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH
        },
        cache_config=CacheConfig(
            use_cache=False,
            write_cache=False
        )
    )
    return get_metrics(result=result)

if __name__ == "__main__" : 
    run()
