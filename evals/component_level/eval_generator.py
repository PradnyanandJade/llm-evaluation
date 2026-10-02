from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
from deepeval.evaluate import evaluate,CacheConfig
from src.config import settings
from src.rag.generation.generator import Generator
from evals.helper import load_goldens,get_metrics

GOLDEN_PATH = "./goldens/generator_goldens.json"
JUDGE_MODEL = settings.OPENAI_MODEL_NAME
THRESHOLD = 0.7
generator = Generator()

def process_context(documents):
    return "\n\n".join(documents)

def run():
    goldens = load_goldens(GOLDEN_PATH)
    test_cases = []
    for g in goldens:
        context = process_context(g["retrieval_context"])
        generated_output = generator.generate(question=g["question"],context=context)
        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=generated_output,
                retrieval_context=g["retrieval_context"]
            )
        )
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD,model=JUDGE_MODEL,include_reason=True)
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
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

if __name__ == '__main__':
    run()