
# LLM Evaluation Pipeline

A production-style evaluation project for RAG applications using **DeepEval** and **LangSmith**.

<img width="1536" height="1024" alt="LLM Evaluation Pipeline for RAG Applications" src="https://github.com/user-attachments/assets/7ad3fd8f-cf0c-4dd8-9ed0-547b667ff5b9" />

## Evaluation Framework

### Evaluation Types
- **Automated** — metric-based evaluation
- **LLM-as-a-Judge** — LLM-based evaluation
- **Human** — human review

### Offline Evaluation
Run against curated datasets before deployment.

- **Component Level** — Retriever, Reranker, Generator
- **Pipeline / Workflow Level** — End-to-end RAG pipeline
- **Application Level**
  - App Quality
  - Safety
  - Operations

### Online Evaluation
Production evaluation using **LangSmith**.

- Capture real application traces
- Sample production runs
- Evaluate with DeepEval metrics
- Attach evaluation feedback to LangSmith runs
- Monitor and improve application quality continuously

## Tech Stack

**Python · LangChain · OpenAI · Hugging Face · DeepEval · LangSmith**

## Project Structure

```text
src/                    # RAG application
evals/
├── offline/            # Offline evaluations
└── online_evals/       # Online evaluation worker
tests/                  # Tests
goldens/                # Evaluation datasets
prompts/                # Evaluation / application prompts
```

## Key Metrics

- Faithfulness
- Answer Relevancy
- Contextual Relevancy
- Contextual Precision
- Contextual Recall
- Quality, Safety & Operational metrics

## Goal

Build a repeatable evaluation workflow to measure **RAG quality, safety, and production performance** before and after deployment.
