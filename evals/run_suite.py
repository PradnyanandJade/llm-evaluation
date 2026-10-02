
# componenent level
from evals.component_level.eval_retriever import run as run_eval_retriever
from evals.component_level.eval_generator import run as run_eval_generator

# pipeline / workflow level
from evals.pipeline_or_workflow_level.eval_rag_pipeline import run as run_eval_rag_pipeline

# application level
# application quality
from evals.application_level.eval_application_quality import run as run_eval_application_quality
# safety
from evals.application_level.eval_safety import run as run_eval_safety
# operations
from evals.application_level.eval_operations import run as run_eval_operations
from evals.helper import save_report


def main():

    print("\n" + "=" * 70)
    print("RUNNING COMPLETE EVALUATION SUITE")
    print("=" * 70)

    # ==============================================================
    # COMPONENT LEVEL
    # ==============================================================

    print("\n")
    print("=" * 70)
    print("COMPONENT LEVEL EVALUATION")
    print("=" * 70)

    print("\n[1/6] Retriever Evaluation")
    retriever_metrics = run_eval_retriever()

    print("\n[2/6] Generator Evaluation")
    generator_metrics = run_eval_generator()

    # ==============================================================
    # PIPELINE / WORKFLOW LEVEL
    # ==============================================================

    print("\n")
    print("=" * 70)
    print("PIPELINE / WORKFLOW LEVEL EVALUATION")
    print("=" * 70)

    print("\n[3/6] RAG Pipeline Evaluation")
    pipeline_metrics = run_eval_rag_pipeline()


    # ==============================================================
    # APPLICATION LEVEL
    # ==============================================================

    print("\n")
    print("=" * 70)
    print("APPLICATION LEVEL EVALUATION")
    print("=" * 70)

    print("\n[4/6] Application Quality Evaluation")
    quality_metrics = run_eval_application_quality()

    print("\n[5/6] Safety Evaluation")
    safety_metrics = run_eval_safety()

    print("\n[6/6] Operations Evaluation")
    operations_metrics = run_eval_operations()

    # ==============================================================
    # COMPLETE
    # ==============================================================

    print("\n")
    print("=" * 70)
    print("COMPLETE EVALUATION SUITE FINISHED")
    print("=" * 70)

    # ==============================================================
    # FINAL REPORT
    # ==============================================================

    final_report = {
        "component": {
            "retriever": retriever_metrics,
            "generator": generator_metrics
        },
        "pipeline": pipeline_metrics,
        "application": {
            "quality": quality_metrics,
            "safety": safety_metrics,
            "operations": operations_metrics
        }
    }

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)

    print(final_report)
    save_report(final_report=final_report)
    return final_report

if __name__ == "__main__":
    main()