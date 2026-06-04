"""
Evaluation helpers shared across all notebooks.

Provides:
  - evaluate_model()          compute accuracy / precision / recall / F1
  - plot_confusion_matrix()   seaborn heatmap, saved to disk as PNG
  - print_metrics_table()     pretty-print a list of result dicts
  - save_results_csv()        persist results list to CSV
  - show_prediction_examples() surface correct and incorrect examples
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# Label names used consistently across all plots and tables
LABEL_NAMES = ['Sad', 'Happy']


# --------------------------------------------------------------------------- #
#  Core metrics                                                                #
# --------------------------------------------------------------------------- #

def evaluate_model(
    model_name: str,
    vectorizer_name: str,
    y_true,
    y_pred,
) -> Dict[str, Any]:
    """
    Compute classification metrics and return them as a dict.

    Returns
    -------
    {
        'model':       str,
        'vectorizer':  str,
        'accuracy':    float,
        'precision':   float,
        'recall':      float,
        'f1':          float,
    }
    """
    return {
        'model':      model_name,
        'vectorizer': vectorizer_name,
        'accuracy':   round(accuracy_score(y_true, y_pred), 4),
        'precision':  round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall':     round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1':         round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }


# --------------------------------------------------------------------------- #
#  Confusion matrix plot                                                       #
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(
    model_name: str,
    vectorizer_name: str,
    y_true,
    y_pred,
    save_dir: str,
    show: bool = False,
) -> None:
    """
    Plot and save a seaborn confusion matrix heatmap.

    File is saved as: <save_dir>/<model_name>_<vectorizer_name>.png
    Spaces in names are replaced with underscores.
    """
    cm = confusion_matrix(y_true, y_pred)
    safe_model = model_name.replace(' ', '_')
    safe_vec   = vectorizer_name.replace(' ', '_')

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
    ax.set_title(f'{model_name}  |  {vectorizer_name}', fontsize=12, pad=10)
    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, f'{safe_model}_{safe_vec}.png')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Tabular output                                                              #
# --------------------------------------------------------------------------- #

def print_metrics_table(results: List[Dict[str, Any]]) -> None:
    """
    Pretty-print a list of evaluate_model() result dicts as a formatted table.
    """
    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)
    col_order = ['model', 'vectorizer', 'accuracy', 'precision', 'recall', 'f1']
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(['vectorizer', 'f1'], ascending=[True, False])

    # Format floats as percentage strings for readability
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f'{x * 100:.2f}%')

    print(df.to_string(index=False))


def save_results_csv(results: List[Dict[str, Any]], path: str) -> None:
    """
    Save a list of result dicts to a CSV file.
    Creates parent directories if they do not exist.
    """
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[evaluation] Results saved to {path}")


# --------------------------------------------------------------------------- #
#  Prediction examples                                                         #
# --------------------------------------------------------------------------- #

def show_prediction_examples(
    texts: List[str],
    y_true,
    y_pred,
    n: int = 5,
    label_map: Dict[int, str] = None,
) -> None:
    """
    Print n correct predictions and n incorrect predictions side by side.

    Parameters
    ----------
    texts     : raw or cleaned text strings (same order as y_true / y_pred)
    y_true    : true labels (array-like)
    y_pred    : predicted labels (array-like)
    n         : how many examples of each type to show
    label_map : optional {0: 'Sad', 1: 'Happy'} mapping; uses int labels if None
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_idx   = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    def _fmt_label(label):
        if label_map:
            return label_map.get(label, str(label))
        return str(label)

    print("=" * 70)
    print(f"CORRECT PREDICTIONS (showing {min(n, len(correct_idx))})")
    print("=" * 70)
    for i in correct_idx[:n]:
        print(f"  Text  : {texts[i][:120]}")
        print(f"  True  : {_fmt_label(y_true[i])}   |   Predicted: {_fmt_label(y_pred[i])}")
        print("-" * 70)

    print()
    print("=" * 70)
    print(f"INCORRECT PREDICTIONS (showing {min(n, len(incorrect_idx))})")
    print("=" * 70)
    for i in incorrect_idx[:n]:
        print(f"  Text  : {texts[i][:120]}")
        print(f"  True  : {_fmt_label(y_true[i])}   |   Predicted: {_fmt_label(y_pred[i])}")
        print("-" * 70)


# --------------------------------------------------------------------------- #
#  Full classification report shortcut                                         #
# --------------------------------------------------------------------------- #

def full_report(model_name: str, vectorizer_name: str, y_true, y_pred) -> None:
    """
    Print sklearn's full classification_report with a header line.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}   |   Vectorizer: {vectorizer_name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
