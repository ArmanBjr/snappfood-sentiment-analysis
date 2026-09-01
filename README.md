# Persian Sentiment Analysis — Snappfood Reviews

Project 3 for **Fundamentals of Computational Intelligence** at Ferdowsi University of Mashhad (FUM). Binary sentiment classification (HAPPY / SAD) on Persian Snappfood reviews using classical ML, ensemble stacking, and ParsBERT fine-tuning.

> Course report: [`report/report.pdf`](report/report.pdf)

**Authors:** AmirHosein Abolfazli · **Arman Bijari** — [ArmanBjr](https://github.com/ArmanBjr)  
**Professor:** Dr. Fazl Ersi  
**Dataset:** [Snappfood Persian Sentiment Analysis (Kaggle)](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis)

---

## Results (test set)

| Phase | Best approach | F1 | Accuracy |
|---|---|---:|---:|
| 3 — Base models | SVM (Linear) + Hybrid vectorizer (tuned) | **0.8574** | **0.8578** |
| 5 — Stacking | StackingClassifier + Hybrid | 0.8578 | 0.8580 |
| 6 — ParsBERT (frozen) | Stacking on ParsBERT embeddings | 0.8451 | 0.8453 |
| 6 — ParsBERT v2 | Fine-tuned Snappfood-BERT (`06_v2.ipynb`) | see notebook | see notebook |

Classical ML + hybrid features outperformed frozen ParsBERT embeddings on this split; fine-tuned BERT is explored in `06_v2.ipynb`.

---

## Quick Start

```bash
git clone https://github.com/ArmanBjr/sentiment-analyze-snappfood.git
cd sentiment-analyze-snappfood

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

Place the raw CSV at `data/raw/snappfood.csv` (not committed — download from Kaggle).

```bash
jupyter lab notebooks/
```

---

## Notebooks (run in order)

| Phase | Notebook | Description |
|---|---|---|
| 1 | `01_preprocessing.ipynb` | Persian text cleaning with Hazm |
| 2 | `02_vectorization.ipynb` | CountVectorizer, TF-IDF, Word2Vec |
| 3 | `03_base_models.ipynb` | NB, SVM, DT, RF, KNN, AdaBoost × vectorizers |
| 4 | `04_vectorizer_selection.ipynb` | Best vectorizer analysis (Hybrid wins) |
| 5 | `05_stacking.ipynb` | StackingClassifier ensemble |
| 6a | `06_parsbert.ipynb` | Frozen ParsBERT embeddings + classical heads |
| 6b | `06_parsbert_bonus.ipynb` | ParsBERT experiments (bonus) |
| 6c | `06_v2.ipynb` | Fine-tuned Snappfood-BERT end-to-end |

A frozen course submission copy lives in `submission/notebooks/`.

---

## Project Structure

```
NLP-Ensemble/
├── notebooks/              # main workflow (phases 1–6)
├── submission/notebooks/   # course submission snapshot
├── src/
│   ├── preprocessing.py    # Persian text cleaning
│   ├── vectorizers.py      # MeanWord2VecVectorizer, hybrid features
│   ├── evaluation.py       # metrics and confusion matrices
│   └── bert_features.py    # ParsBERT embedding extraction
├── scripts/
│   ├── predict_one.py      # single-review inference helper
│   └── *.sh                # WSL/GPU run helpers for phase 6 v2
├── outputs/
│   ├── figures/
│   ├── results/            # CSV/JSON metrics per phase
│   ├── confusion_matrices/
│   └── v2_parsbert/        # v2 splits + cleaned data (weights gitignored)
└── report/
    └── report.pdf
```

---

## Phases Overview

1. **Preprocessing** — Hazm normalization, tokenization, lemmatization  
2. **Vectorization** — CountVectorizer, TF-IDF, Word2Vec (trained from scratch), Hybrid  
3. **Base Models** — 6 classifiers × multiple vectorizers  
4. **Vectorizer Selection** — Hybrid vectorizer selected (peak F1 0.8574)  
5. **Stacking** — StackingClassifier; marginal gain over best single model (+0.0004 F1)  
6. **ParsBERT** — frozen embeddings and fine-tuned Snappfood-BERT (`06_v2.ipynb`)

---

## Authors & License

**AmirHosein Abolfazli** · **Arman Bijari** — [ArmanBjr](https://github.com/ArmanBjr)

Released under the [MIT License](LICENSE).
