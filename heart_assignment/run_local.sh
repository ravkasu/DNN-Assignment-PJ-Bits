#!/bin/bash
timestamp(){ date +"%Y-%m-%d %H:%M:%S"; }
echo ">>> START: $(timestamp)"

echo ">>> Activating virtual environment..."
source .venv/bin/activate

export PYTHONPATH="$PWD"

echo "[$(timestamp)] Step 1: Preparing processed dataset..."
python src/data_loader.py || { echo "data prep failed"; exit 1; }

echo "[$(timestamp)] Step 2: Running Baseline Model..."
python src/baseline.py || { echo "baseline failed"; exit 1; }

echo "[$(timestamp)] Step 3: Running MLP Model..."
python src/mlp.py || { echo "mlp failed"; exit 1; }

echo "[$(timestamp)] Step 4: Showing Saved Results..."
python src/results_extractor.py || { echo "results extraction failed"; exit 1; }

echo ">>> DONE: $(timestamp)"
