#!/usr/bin/env bash
set -euo pipefail

# Activate the GPU Python env (adjust if you use a different venv)
if [[ -f "$HOME/tf-gpu/bin/activate" ]]; then
  source "$HOME/tf-gpu/bin/activate"
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
  source "$HOME/miniconda3/bin/activate"
fi

echo "=== Python ==="
which python
python --version

echo ""
echo "=== NVIDIA ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== PyTorch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Key packages ==="
python - <<'PY'
pkgs = ["transformers", "accelerate", "sklearn", "pandas", "hazm", "matplotlib", "seaborn"]
for p in pkgs:
    try:
        m = __import__(p)
        print(f"{p}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{p}: MISSING")
PY
