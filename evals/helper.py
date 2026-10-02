import json
from datetime import datetime
import json
from pathlib import Path

def load_goldens(GOLDEN_PATH:str):
    with open(GOLDEN_PATH,"r") as f:
        return json.load(f)

def get_metrics(result):
    metrics = {}
    for test_result in result.test_results:
        for metric in test_result.metrics_data:
            metrics.setdefault(metric.name, []).append(metric.score)
    return {
        name: sum(scores) / len(scores)
        for name, scores in metrics.items()
    }

def save_report(final_report):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("evaluation_runs").mkdir(exist_ok=True)
    file_path = f"evaluation_runs/eval_{timestamp}.json"
    with open(file_path, "w") as f:
        json.dump(final_report, f, indent=4)
    print(f"\nEvaluation results saved to: {file_path}")