# Project quick-run (one-shot commands you can paste into your VS Code terminal)

Copy these commands/blocks **in order** from the project root (`heart_assignment`) — they will create the environment, produce synthetic data, create the required scripts, run the pipeline and show results. Each step is a single pasteable command or one small heredoc block.
If you already created files earlier, the blocks will overwrite them with the working versions used during our session.

> **Screenshot used in troubleshooting:** `/mnt/data/Screenshot 2025-11-20 at 1.31.51 PM.png`

---

## 0. Prerequisites (macOS / Linux)

```bash
# ensure Python3 and git are installed (run manually if needed)
python3 --version
```

---

## 1. Create & activate venv (run once)

```bash
python3 -m venv .venv && source .venv/bin/activate
```

---

## 2. Install dependencies

```bash
pip install --upgrade pip && pip install -r requirements.txt || pip install numpy pandas scikit-learn joblib matplotlib jupyter pytest
```

---

## 3. Add PYTHONPATH (project .env + VSCode setting)

Create `.env`:

```bash
echo 'PYTHONPATH=${PWD}' > .env
```

(optional) create VSCode settings so editor picks `.env`:

```bash
mkdir -p .vscode && cat > .vscode/settings.json <<'JSON'
{
  "python.envFile": "${workspaceFolder}/.env"
}
JSON
```

---

## 4. Generate synthetic dataset (creates `data/raw/heart.csv`)

```bash
mkdir -p data/raw data/processed results && python - <<'PY'
import numpy as np, pandas as pd, os
np.random.seed(0)
n=300
age = np.random.randint(29,77,size=n)
sex = np.random.randint(0,2,size=n)
cp = np.random.randint(0,4,size=n)
trestbps = np.random.randint(94,200,size=n)
chol = np.random.randint(126,564,size=n)
fbs = np.random.randint(0,2,size=n)
restecg = np.random.randint(0,2,size=n)
thalach = np.random.randint(71,202,size=n)
exang = np.random.randint(0,2,size=n)
oldpeak = np.round(np.random.uniform(0.0,6.2,size=n),2)
slope = np.random.randint(0,3,size=n)
ca = np.random.randint(0,4,size=n)
thal = np.random.randint(0,3,size=n)
target = ((age>50) & (chol>240)) | (cp>2)
target = target.astype(int)
df = pd.DataFrame(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                       fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                       oldpeak=oldpeak, slope=slope, ca=ca, thal=thal, target=target))
df.to_csv(os.path.join('data','raw','heart.csv'), index=False)
print('WROTE data/raw/heart.csv rows:', len(df))
PY
```

---

## 5. Create repo Python scripts (overwrite if present)

### `src/data_loader.py`

```bash
mkdir -p src && cat > src/data_loader.py <<'PY'
import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_RAW = os.path.join("data", "raw", "heart.csv")
DATA_PROCESSED_X_TRAIN = os.path.join("data", "processed", "X_train.csv")
DATA_PROCESSED_Y_TRAIN = os.path.join("data", "processed", "y_train.csv")
DATA_PROCESSED_X_TEST = os.path.join("data", "processed", "X_test.csv")
DATA_PROCESSED_Y_TEST = os.path.join("data", "processed", "y_test.csv")

def load_raw_dataframe(path=DATA_RAW):
    df = pd.read_csv(path)
    return df

def prepare_and_save_test_train(test_size=0.2, random_state=42):
    df = load_raw_dataframe()
    if "target" not in df.columns:
        raise ValueError("No 'target' column in raw data")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train.to_csv(DATA_PROCESSED_X_TRAIN, index=False)
    X_test.to_csv(DATA_PROCESSED_X_TEST, index=False)
    y_train.to_csv(DATA_PROCESSED_Y_TRAIN, index=False)
    y_test.to_csv(DATA_PROCESSED_Y_TEST, index=False)
    print("Processed data saved to data/processed/")
    return True

if __name__ == "__main__":
    prepare_and_save_test_train()
PY
```

### `src/utils.py`

```bash
cat > src/utils.py <<'PY'
import os, joblib, json

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
PY
```

### `src/metrics.py`

```bash
cat > src/metrics.py <<'PY'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
PY
```

### `src/baseline.py`

```bash
cat > src/baseline.py <<'PY'
import os, pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils import save_model, save_json
from src.metrics import classification_metrics
from src.data_loader import prepare_and_save_test_train, DATA_PROCESSED_X_TEST, DATA_PROCESSED_Y_TEST, DATA_PROCESSED_X_TRAIN, DATA_PROCESSED_Y_TRAIN

MODEL_PATH = os.path.join("results", "baseline_model.joblib")
METRICS_PATH = os.path.join("results", "baseline_metrics.json")

def run_baseline():
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
    save_model(model, MODEL_PATH)
    save_json(metrics, METRICS_PATH)
    print("BASELINE TRAINED")
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    run_baseline()
PY
```

### `src/mlp.py`

```bash
cat > src/mlp.py <<'PY'
import os, pandas as pd
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
PY
```

### `src/results_extractor.py`

```bash
cat > src/results_extractor.py <<'PY'
import os
from src.utils import load_json
def show_results():
    for fname in ["results/baseline_metrics.json", "results/mlp_metrics.json"]:
        if os.path.exists(fname):
            print("===", fname)
            print(load_json(fname))
        else:
            print("Missing:", fname)
    print("\nFiles in results/")
    for f in sorted(os.listdir("results")):
        print(" -", f)
if __name__ == '__main__':
    show_results()
PY
```

---

## 6. Create `run_local.sh` (pipeline runner) and make it executable

```bash
cat > run_local.sh <<'SH'
#!/bin/bash
echo ">>> Activating virtual environment..."
source .venv/bin/activate
export PYTHONPATH="$PWD"
echo ">>> Step 1: Preparing processed dataset..."
python src/data_loader.py
echo ">>> Step 2: Running Baseline Model..."
python src/baseline.py
echo ">>> Step 3: Running MLP Model..."
python src/mlp.py
echo ">>> Step 4: Showing Saved Results..."
python src/results_extractor.py
echo ">>> DONE."
SH
chmod +x run_local.sh
```

---

## 7. (Optional) create `run.sh` that calls `run_local.sh`

```bash
cat > run.sh <<'SH'
#!/usr/bin/env bash
set -e
if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; else echo ".venv not found; activate env manually."; fi
export PYTHONPATH="$PWD"
./run_local.sh
SH
chmod +x run.sh
```

---

## 8. Run the full pipeline (one-line)

```bash
./run_local.sh
# or
./run.sh
```

Expected console output: dataset saved → BASELINE TRAINED + metrics → MLP TRAINED + metrics → results printed.

---

## 9. Run tests (simple pytest)

```bash
cat > tests/test_results_exist.py <<'PY'
def test_results_exist():
    import os
    assert os.path.exists("results/baseline_metrics.json")
    assert os.path.exists("results/mlp_metrics.json")
PY
pip install pytest -q || true
pytest -q
```

---

## 10. Inspect results & saved models

```bash
ls -l results
cat results/baseline_metrics.json
cat results/mlp_metrics.json
```

---

## 11. (Optional) Freeze exact dependencies used

```bash
pip freeze > requirements.txt
```

---

## 12. Clean up (optional)

```bash
# remove generated artifacts
rm -rf data/processed results run_local.sh run.sh tests src/*.py
```

---

## Notes & Tips

* Always run the commands from the project root (folder containing `run_local.sh`, `src/`, `.venv`, `data/`).
* If you open VS Code, ensure it loads `.env` (see `.vscode/settings.json`) or set `PYTHONPATH` manually in terminal:
  `export PYTHONPATH="$PWD"`
* If a script silently exits with no output, re-run with `-u` / `-X faulthandler` or run the script file through `python -m pdb` to debug.

----------------------------------------------------------------------------------------
