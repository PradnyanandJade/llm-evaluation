import json
import time
import statistics
import numpy as np
from src.rag.pipeline import RAGPipeline
from evals.helper import load_goldens,get_metrics

GOLDEN_PATH = "./goldens/pipeline_goldens.json"

pipeline = RAGPipeline()


def run():
    goldens = load_goldens(GOLDEN_PATH)
    latencies = []
    results = []
    for golden in goldens:
        question = golden["question"]
        start = time.perf_counter()
        result = pipeline.run(
            question=question
        )
        end = time.perf_counter()
        latency = end - start
        latencies.append(latency)
        results.append({
            "id": golden["id"],
            "question": question,
            "latency_seconds": latency,
            "blocked": result["blocked"]
        })

    average = statistics.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\nLATENCY EVALUATION")
    print("=" * 50)

    print(f"Requests: {len(latencies)}")
    print(f"Average:  {average:.3f}s")
    print(f"P50:      {p50:.3f}s")
    print(f"P95:      {p95:.3f}s")
    print(f"P99:      {p99:.3f}s")

    print("\nPer Request")
    print("=" * 50)

    for result in results:
        print(
            f"{result['id']} | "
            f"{result['latency_seconds']:.3f}s"
        )
        
    return {
        # "requests": len(latencies),
        "average_latency": average,
        "p50": p50,
        "p95": p95,
        "p99": p99
    }


if __name__ == "__main__":
    run()