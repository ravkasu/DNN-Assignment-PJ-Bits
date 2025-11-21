import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils import save_model, save_json
from src.metrics import classification_metrics
from src.data_loader import prepare_and_save_test_train, DATA_PROCESSED_X_TEST, DATA_PROCESSED_Y_TEST, DATA_PROCESSED_X_TRAIN, DATA_PROCESSED_Y_TRAIN

MODEL_PATH = os.path.join("results", "baseline_model.joblib")
METRICS_PATH = os.path.join("results", "baseline_metrics.json")

def run_baseline():
    # ensure processed data exists
    if not os.path.exists(DATA_PROCESSED_X_TRAIN):
        prepare_and_save_test_train()
    X_train = pd.read_csv(DATA_PROCESSED_X_TRAIN)
    y_train = pd.read_csv(DATA_PROCESSED_Y_TRAIN).squeeze()
    X_test = pd.read_csv(DATA_PROCESSED_X_TEST)
    y_test = pd.read_csv(DATA_PROCESSED_Y_TEST).squeeze()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = classification_metrics(y_test, preds)

    # save model and metrics
    save_model(model, MODEL_PATH)
    save_json(metrics, METRICS_PATH)

    # print a friendly summary
    print("BASELINE TRAINED")
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    run_baseline()
