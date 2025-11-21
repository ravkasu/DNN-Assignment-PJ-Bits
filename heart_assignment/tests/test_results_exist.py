def test_results_exist():
    import os
    assert os.path.exists("results/baseline_metrics.json")
    assert os.path.exists("results/mlp_metrics.json")
