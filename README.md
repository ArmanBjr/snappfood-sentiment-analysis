# Persian Sentiment Analysis — NLP + Ensemble Learning
## Computational Intelligence Project 3 | Ferdowsi University of Mashhad

**Students:** AmirHosein Abolfazli (4022262035) · Arman Bijari (4021262131)  
**Professor:** Dr. Fazl-Ersi  
**Dataset:** [Snappfood Persian Sentiment Analysis](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis)

---

## Project Structure

```
NLP-Ensemble/
├── data/
│   ├── raw/            # Original dataset CSV (place snappfood.csv here)
│   └── processed/      # Cleaned/preprocessed data saved as CSV
├── notebooks/
│   ├── 01_preprocessing.ipynb       # Phase 1: text cleaning
│   ├── 02_vectorization.ipynb       # Phase 2: CountVec, TF-IDF, Word2Vec
│   ├── 03_base_models.ipynb         # Phase 3: 6 models × 3 vectorizers
│   ├── 04_vectorizer_selection.ipynb # Phase 4: best vectorizer analysis
│   └── 05_stacking.ipynb            # Phase 5: StackingClassifier
├── src/
│   ├── preprocessing.py   # Persian text cleaning functions
│   ├── vectorizers.py     # MeanWord2VecVectorizer class
│   └── evaluation.py      # Metrics, confusion matrix helpers
├── models/
│   └── word2vec/          # Saved gensim Word2Vec model
├── outputs/
│   ├── figures/           # Plots (word clouds, distributions)
│   ├── results/           # CSV result tables per phase
│   └── confusion_matrices/ # Saved confusion matrix PNGs
├── report/                # LaTeX report files
└── README.md
```

## Setup

```bash
pip install hazm gensim scikit-learn pandas numpy matplotlib seaborn joblib
```

## Phases

1. **Preprocessing** — Hazm normalization, tokenization, lemmatization
2. **Vectorization** — CountVectorizer, TF-IDF, Word2Vec (trained from scratch)
3. **Base Models** — NB, SVM, DT, RF, KNN, AdaBoost
4. **Best Vectorizer** — Analysis and selection
5. **Stacking** — StackingClassifier with best vectorizer
