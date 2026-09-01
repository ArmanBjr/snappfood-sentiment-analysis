#!/usr/bin/env bash
# Run Phase 6 v2 notebook end-to-end from WSL (Ubuntu).
set -euo pipefail

REPO_WIN='E:/University/University_Subjects/6th/Computational_Intelligence/Projects/3/NLP-Ensemble'
REPO="/mnt/e/University/University_Subjects/6th/Computational_Intelligence/Projects/3/NLP-Ensemble"

if [[ ! -d "$REPO" ]]; then
  echo "Repo not found at $REPO"
  echo "Edit REPO in this script if your mount path differs."
  exit 1
fi

if [[ -f "$HOME/tf-gpu/bin/activate" ]]; then
  source "$HOME/tf-gpu/bin/activate"
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
  source "$HOME/miniconda3/bin/activate"
fi

cd "$REPO/notebooks"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

# Ensure Phase 1 outputs exist
if [[ ! -f "$REPO/data/raw/Snappfood - Sentiment Analysis.csv" ]]; then
  echo ""
  echo "Missing data/raw/Snappfood - Sentiment Analysis.csv"
  exit 1
fi

# Optional deps for 06_v2 pipeline
python -c "import hazm" 2>/dev/null || pip install -q hazm
python -c "import tqdm" 2>/dev/null || pip install -q tqdm
python -c "import jupyter" 2>/dev/null || pip install -q jupyter nbconvert ipykernel

echo ""
echo "Executing notebooks/06_v2.ipynb (GPU training — several hours)..."
jupyter nbconvert --to notebook --execute 06_v2.ipynb \
  --output 06_v2_executed.ipynb \
  --ExecutePreprocessor.timeout=-1

echo ""
echo "Done. Outputs saved under outputs/results/ and outputs/confusion_matrices/"
