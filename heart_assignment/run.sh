#!/usr/bin/env bash
set -e

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo ".venv not found; activate your environment manually."
fi

export PYTHONPATH="$PWD"

# call pipeline
./run_local.sh
