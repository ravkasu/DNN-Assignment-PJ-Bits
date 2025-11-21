# src/results_extractor.py
import json
import os

_TEMPLATE = {
    "dataset_name": "PUT_DATASET_NAME_HERE",
    "n_samples": 0,
    "n_features": 0,
    "problem_type": "regression",  # 'regression' | 'binary_classification' | 'multi_class'
    "primary_metric": "rmse",
    "baseline_model": {
        "test_rmse": None,
        "test_mse": None,
        "test_mae": None,
        "training_time_seconds": None
    },
    "mlp_model": {
        "architecture": [],
        "test_rmse": None,
        "test_mse": None,
        "test_mae": None,
        "training_time_seconds": None
    }
}

_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "submission_results.json")

def get_assignment_results():
    """
    Returns a dictionary used by the auto-grader.
    Priority:
    1) If submission_results.json exists, return its contents (recommended).
    2) Otherwise return the template dict (you must replace values before final submission).
    """
    try:
        json_path = os.path.abspath(_JSON_PATH)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            return data
    except Exception:
        # fallthrough to template
        pass
    return _TEMPLATE.copy()
