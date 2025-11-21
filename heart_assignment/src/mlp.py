import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.utils import save_model, save_json
from src.metrics import classification_metrics
from src.data_loader import prepare_and_save_test_train, DATA_PROCESSED_X_TRAIN, DATA_PROCESSED_Y_TRAIN, DATA_PROCESSED_X_TEST, DATA_PROCESSED_Y_TEST

MODEL_PATH = os.path.join("results", "mlp_model.joblib")
METRICS_PATH = os.path.join("results", "mlp_metrics.json")

def run_mlp(hidden_layer_sizes=(50,), max_iter=200):
    if not os.path.exists(DATA_PROCESSED_X_TRAIN):
        prepare_and_save_test_train()
    X_train = pd.read_csv(DATA_PROCESSED_X_TRAIN)
    y_train = pd.read_csv(DATA_PROCESSED_Y_TRAIN).squeeze()
    X_test = pd.read_csv(DATA_PROCESSED_X_TEST)
    y_test = pd.read_csv(DATA_PROCESSED_Y_TEST).squeeze()

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    metrics = classification_metrics(y_test, preds)

    save_model(clf, MODEL_PATH)
    save_json(metrics, METRICS_PATH)

    print("MLP TRAINED")
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    run_mlp()
