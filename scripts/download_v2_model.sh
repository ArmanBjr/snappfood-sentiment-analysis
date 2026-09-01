#!/usr/bin/env bash
# Download HooshvareLab Snappfood BERT via HF mirror (works better from Iran / slow links).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
LOCAL_DIR="${REPO_ROOT}/models/bert-fa-base-uncased-sentiment-snappfood"

mkdir -p "$LOCAL_DIR"

# Mirror endpoint — use official hub if you have stable access:
#   unset HF_ENDPOINT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -f "${LOCAL_DIR}/pytorch_model.bin" ]] || [[ -f "${LOCAL_DIR}/model.safetensors" ]]; then
  echo "Model already present in ${LOCAL_DIR}"
  ls -lh "${LOCAL_DIR}"
  exit 0
fi

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "Downloading ${MODEL_ID} -> ${LOCAL_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --local-dir "$LOCAL_DIR"
elif python -c "import huggingface_hub" 2>/dev/null; then
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${LOCAL_DIR}",
    resume_download=True,
)
PY
else
  echo "Install huggingface_hub: pip install huggingface_hub"
  exit 1
fi

echo "Done."
ls -lh "${LOCAL_DIR}"
