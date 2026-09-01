#!/usr/bin/env bash
# Quick smoke test: paths, data, imports (no GPU training).
set -euo pipefail

REPO="/mnt/e/University/University_Subjects/6th/Computational_Intelligence/Projects/3/NLP-Ensemble"
source "$HOME/tf-gpu/bin/activate"
cd "$REPO/notebooks"

python - <<'PY'
import os, sys
REPO_ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
DATA_PROC = os.path.join(REPO_ROOT, 'data', 'processed')

import pandas as pd
import numpy as np
import torch
from bert_features import DEFAULT_MODEL_NAME
from evaluation import evaluate_model
from vectorizers import build_word_char_union

train_df = pd.read_csv(os.path.join(DATA_PROC, 'train.csv'))
print('train rows:', len(train_df))
print('model:', DEFAULT_MODEL_NAME)
print('cuda:', torch.cuda.is_available())
print('imports OK')
PY
