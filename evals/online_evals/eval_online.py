# evals/online/online_worker.py

import time
from dotenv import load_dotenv
from langsmith import Client

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.config import settings


load_dotenv()

PROJECT = settings.LANGSMITH_PROJECT
JUDGE_MODEL = settings.OPENAI_MODEL_NAME
THRESHOLD = 0.7
SAMPLE_RATE = 1.0
POLL_SECONDS = 60

client = Client(
    api_key=settings.LANGSMITH_API_KEY,
    api_url=settings.LANGSMITH_ENDPOINT,
)


def sampled(run):
    return hash(str(run.id)) % 100 < SAMPLE_RATE * 100


def existing_keys(run):
    feedback = client.list_feedback(run_ids=[run.id])
    return {item.key for item in feedback}


def score_recent_traces():

    print("\nChecking LangSmith traces...")

    runs = list(
        client.list_runs(
            project_name=PROJECT,
            filter='eq(name, "RAG Pipeline")',
        )
    )

    print(f"Project: {PROJECT}")
    print(f"Runs found: {len(runs)}")

    for run in runs:

        print(f"\nProcessing run: {run.id}")
        print(f"Run name: {run.name}")

        if not sampled(run):
            print("Skipped by sampling")
            continue

        inputs = run.inputs or {}
        outputs = run.outputs or {}

        question = inputs.get("question")
        answer = outputs.get("answer")
        documents = outputs.get("documents", [])

        print(f"Question: {question}")
        print(f"Answer exists: {bool(answer)}")
        print(f"Documents: {len(documents)}")

        if not question or not answer or not documents:
            print("Skipping: missing question, answer, or documents")
            continue

        # LangSmith serializes your LangChain Documents
        # with page_content.
        context = [
            document.get("page_content", "")
            if isinstance(document, dict)
            else document.page_content
            for document in documents
        ]

        # Remove empty context items
        context = [
            item for item in context
            if item.strip()
        ]

        if not context:
            print("Skipping: empty retrieval context")
            continue

        already = existing_keys(run)

        print(f"Existing feedback: {already}")

        jobs = [

            (
                "faithfulness",
                FaithfulnessMetric(
                    threshold=THRESHOLD,
                    model=JUDGE_MODEL,
                    include_reason=True,
                ),
                {
                    "input": question,
                    "actual_output": answer,
                    "retrieval_context": context,
                },
            ),

            (
                "answer_relevancy",
                AnswerRelevancyMetric(
                    threshold=THRESHOLD,
                    model=JUDGE_MODEL,
                    include_reason=True,
                ),
                {
                    "input": question,
                    "actual_output": answer,
                },
            ),

            (
                "contextual_relevancy",
                ContextualRelevancyMetric(
                    threshold=THRESHOLD,
                    model=JUDGE_MODEL,
                    include_reason=True,
                ),
                {
                    "input": question,
                    "actual_output": answer,
                    "retrieval_context": context,
                },
            ),
        ]

        for key, metric, test_case_kwargs in jobs:

            if key in already:
                print(f"{key}: already evaluated")
                continue

            try:

                print(f"Evaluating {key}...")

                test_case = LLMTestCase(
                    **test_case_kwargs
                )

                metric.measure(test_case)

                print(
                    f"{key} score: {metric.score}"
                )

                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )

                print(
                    f"{key}: feedback added to LangSmith"
                )

            except Exception as e:

                print(
                    f"[{key}] failed for run {run.id}: {e}"
                )


if __name__ == "__main__":

    while True:

        score_recent_traces()

        time.sleep(POLL_SECONDS)