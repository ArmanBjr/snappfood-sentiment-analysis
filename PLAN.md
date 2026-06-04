# PLAN.md — Persian Sentiment Analysis: NLP + Ensemble Learning
## Project 3 | Computational Intelligence | Ferdowsi University of Mashhad

**Read this file at the start of every session. It contains everything needed to continue without re-explanation.**

---

## 1. Project Identity

| Field | Value |
|---|---|
| Course | مبانی هوش محاسباتی (Computational Intelligence) |
| Professor | دکتر فضل ارثی (Dr. Fazl-Ersi) |
| Student 1 | AmirHosein Abolfazli — ID: 4022262035 |
| Student 2 | Arman Bijari — ID: 4021262131 |
| Emails | abolfazli2035@gmail.com / armanbijari5@gmail.com |
| Semester | نیمسال دوم ۱۴۰۵–۱۴۰۴ |
| Submission | One zip named after student IDs, uploaded by one member |
| Plagiarism | Any unusual similarity to other groups = zero grade |
| Bonus | Clean LaTeX report gets extra credit |

---

## 2. Dataset

- **Name:** Snappfood Persian Sentiment Analysis
- **Kaggle:** https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis/data
- **Size:** 70,000 reviews — perfectly balanced
  - 35,000 labeled `Happy` (positive)
  - 35,000 labeled `Sad` (negative)
- **Columns:** `comment` (Persian text), `label` (Happy / Sad)
- **Source:** Snapp!food — Iran's leading online food delivery platform
- **Place raw CSV at:** `data/raw/snappfood.csv`
- **Recommended split:** 80% train / 20% test, stratified, `random_state=42`

---

## 3. Project Phases (what the brief requires)

### Phase 1 — Preprocessing
- Preprocess all Persian text so it is clean and noise-free for NLP models
- Split dataset into train/test with an appropriate ratio (your choice)
- **Report must contain:**
  - Description of each preprocessing step and why it was chosen
  - Several before/after examples showing the effect of preprocessing

### Phase 2 — Vectorization (Feature Extraction)
- Implement and compare three text-to-vector methods:
  1. `CountVectorizer`
  2. `TF-IDF`
  3. `Word2Vec`
- **Report must contain:**
  - Settings used for each vectorizer and the reason for those choices

### Phase 3 — Base Model Training
- Train **at least 5** classical ML models on all three vectorizations
- Suggested models: Naive Bayes, SVM, Decision Tree, Random Forest, KNN, AdaBoost/GBM
- **Report must contain:**
  - Key hyperparameters per model
  - Comparison table: all models × all vectorizers
  - Accuracy, Precision, Recall, F1 for every model
  - Confusion matrix for every model
  - Analysis: which models performed best and why
  - Examples of correct and incorrect predictions from the best model

### Phase 4 — Best Vectorizer Selection
- Analyze Phase 3 results and pick the best vectorization method for Phase 5
- **Report must contain:**
  - Best model+vectorizer combination
  - Best vectorizer selected for next phase
  - Analysis: why this vectorizer outperformed the others

### Phase 5 — Stacking Ensemble
- Using the best vectorizer from Phase 4, combine multiple base models via Stacking
- Goal: determine whether the Stacking ensemble beats the best single model
- **Report must contain:**
  - Which base models were selected for Stacking and why
  - Performance comparison: Stacking vs. best single model
  - Confusion matrix for the Stacking model
  - Accuracy, Precision, Recall, F1 for the Stacking model
  - Correct and incorrect prediction examples

---

## 4. Deliverables

| Item | Description |
|---|---|
| `notebooks/01_preprocessing.ipynb` | Phase 1 |
| `notebooks/02_vectorization.ipynb` | Phase 2 |
| `notebooks/03_base_models.ipynb` | Phase 3 |
| `notebooks/04_vectorizer_selection.ipynb` | Phase 4 |
| `notebooks/05_stacking.ipynb` | Phase 5 |
| `src/preprocessing.py` | Reusable text cleaning functions |
| `src/vectorizers.py` | `MeanWord2VecVectorizer` sklearn transformer |
| `src/evaluation.py` | Metrics + confusion matrix helpers |
| `report/*.tex` | LaTeX Persian-language report |
| Zip file | Named `4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.zip` |

**Language rules:**
- Notebook markdown cells and code comments → **English**
- LaTeX report → **Persian (Farsi)**

---

## 5. Folder Structure

```
NLP-Ensemble/
├── PLAN.md                          ← this file
├── README.md
├── data/
│   ├── raw/
│   │   └── snappfood.csv            ← place dataset here
│   └── processed/
│       ├── train.csv                ← Phase 1 output
│       ├── test.csv                 ← Phase 1 output
│       ├── X_train_count.npz        ← Phase 2 output (scipy sparse)
│       ├── X_test_count.npz
│       ├── X_train_tfidf.npz
│       ├── X_test_tfidf.npz
│       ├── X_train_w2v.npy          ← Phase 2 output (numpy dense)
│       ├── X_test_w2v.npy
│       ├── y_train.npy
│       └── y_test.npy
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_vectorization.ipynb
│   ├── 03_base_models.ipynb
│   ├── 04_vectorizer_selection.ipynb
│   └── 05_stacking.ipynb
├── src/
│   ├── preprocessing.py
│   ├── vectorizers.py
│   └── evaluation.py
├── models/
│   └── word2vec/
│       └── persian_w2v.model        ← Phase 2 output (gensim)
├── outputs/
│   ├── figures/                     ← word clouds, distributions, bar charts
│   ├── results/
│   │   ├── phase3_results.csv       ← model × vectorizer metrics table
│   │   └── best_vectorizer.json     ← Phase 4 decision
│   └── confusion_matrices/          ← PNG per model×vectorizer
└── report/
    ├── 4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.tex
    ├── writing-style.txt            ← tone/style guide for the Persian report
    ├── assets/
    │   └── fum-logo.png
    └── fonts/
        ├── Vazir.ttf
        └── Vazir-Bold.ttf
```

---

## 6. Implementation Plan (Phase by Phase)

### Phase 0 — src/ Modules (always implement first)

These are shared utilities imported by all notebooks.

**`src/preprocessing.py`**
- `clean_text(text: str) -> str` — full pipeline: normalize → clean → tokenize → stopwords → lemmatize → join
- `preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame` — applies `clean_text` to `comment` column, returns df with `cleaned` column
- Uses: `hazm.Normalizer`, `hazm.WordTokenizer`, `hazm.Lemmatizer`, `hazm.stopwords_list`

**`src/vectorizers.py`**
- `MeanWord2VecVectorizer(BaseEstimator, TransformerMixin)`
  - `__init__(self, model, vector_size=100)`
  - `fit(X, y=None) -> self` — no-op
  - `transform(X) -> np.ndarray` — average word vectors per doc; zero vector for fully-OOV docs

**`src/evaluation.py`**
- `evaluate_model(name, vectorizer_name, y_true, y_pred) -> dict` — returns dict with accuracy, precision, recall, f1
- `plot_confusion_matrix(name, vectorizer_name, y_true, y_pred, save_dir)` — seaborn heatmap, saves PNG
- `print_metrics_table(results_list)` — pretty-prints a comparison table from list of dicts
- `save_results_csv(results_list, path)` — saves to CSV

---

### Phase 1 — Notebook 01: Preprocessing

**Steps in notebook:**
1. Load `data/raw/snappfood.csv`
2. EDA: label distribution (bar chart), text length histogram, 5 raw sample reviews
3. Import and call `preprocess_dataframe()` from `src/preprocessing.py`
4. Show before/after table (5 examples minimum)
5. Train/test split: 80/20, `stratify=y`, `random_state=42`
6. Save `data/processed/train.csv`, `data/processed/test.csv`

**Preprocessing steps (in order):**
1. Remove URLs **first** (before normalizer, which would strip `://` and break the URL regex): `re.sub(r'https?://\S+|www\.\S+', '', text)`
2. Hazm `Normalizer().normalize()` — unifies Arabic/Persian char variants, fixes ZWNJ, removes diacritics
3. Remove emojis: `re.sub(r'[^\w\s؀-ۿ]', ' ', text)`
4. Remove digits (Arabic-Indic + Western): `re.sub(r'[۰-۹0-9٠-٩]', '', text)`
5. Remove punctuation: `re.sub(r'[!؟،؛.,:;()\[\]{}\-_"\'«»…]', ' ', text)`
6. Normalize repeated chars: `re.sub(r'(.)\1{2,}', r'\1', text)` — خوووب → خوب (collapse to 1, not 2)
7. Hazm `WordTokenizer().tokenize()`
8. Remove stopwords: `[t for t in tokens if t not in hazm.stopwords_list()]`
9. Hazm lemmatize: `lemmatizer.lemmatize(t).split('#')[0]`
10. Remove tokens shorter than 2 characters
11. Join: `' '.join(tokens)`

---

### Phase 2 — Notebook 02: Vectorization

**CountVectorizer settings:**
```python
CountVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
```
Reason: captures unigrams and bigrams (e.g., خیلی خوب = "very good"); `min_df=2` removes hapax legomena.

**TF-IDF settings:**
```python
TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
```
Reason: `sublinear_tf=True` applies log(1+tf) — reduces dominance of high-frequency terms; bigrams capture sentiment phrases.

**Word2Vec settings (trained from scratch via gensim):**
```python
Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2,
         sg=1, workers=4, epochs=15)
```
- `sg=1` = Skip-gram (better for small corpora than CBOW)
- `vector_size=100` — sufficient for 70k corpus; 300 adds cost without benefit at this scale
- `min_count=2` — ignore words appearing only once
- Document vector = mean of all token vectors in the document
- OOV tokens → zero vector contribution (skipped)
- Save model to `models/word2vec/persian_w2v.model`

**Saved feature matrices:**
- CountVectorizer + TF-IDF → `scipy.sparse.save_npz()` (sparse, efficient)
- Word2Vec → `np.save()` (dense 2D array)
- Labels → `np.save()` for `y_train` and `y_test`

---

### Phase 3 — Notebook 03: Base Models

**6 models to train:**

| Model | Config | Notes |
|---|---|---|
| `ComplementNB` | `alpha=0.1` | Better than MultinomialNB for balanced text; sparse features only |
| `GaussianNB` | default | For Word2Vec features (accepts negatives); dense features only |
| `LinearSVC` wrapped in `CalibratedClassifierCV` | `C=1.0, max_iter=2000` | Fast linear SVM; wrapping adds `predict_proba` for Stacking |
| `DecisionTreeClassifier` | `max_depth=20, random_state=42` | Interpretable baseline |
| `RandomForestClassifier` | `n_estimators=200, n_jobs=-1, random_state=42` | Ensemble baseline |
| `KNeighborsClassifier` | `n_neighbors=7` | Often weak on sparse; included for comparison |
| `AdaBoostClassifier` | `n_estimators=100, random_state=42` | Boosting baseline |

**NB compatibility rule:**
- CountVectorizer + TF-IDF → use `ComplementNB`
- Word2Vec → use `GaussianNB` (MultinomialNB rejects negative float values)

**Experiment loop structure:**
```python
vectorizers = {'CountVectorizer': ..., 'TF-IDF': ..., 'Word2Vec': ...}
models = {'NB': ..., 'SVM': ..., 'DT': ..., 'RF': ..., 'KNN': ..., 'AdaBoost': ...}
results = []
for vec_name, (X_tr, X_te) in vectorizers.items():
    for model_name, model in models.items():
        # handle NB special case
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        results.append(evaluate_model(model_name, vec_name, y_test, y_pred))
        plot_confusion_matrix(model_name, vec_name, y_test, y_pred, save_dir)
```

**Save:** `outputs/results/phase3_results.csv`

---

### Phase 4 — Notebook 04: Vectorizer Selection

**Steps:**
1. Load `phase3_results.csv`
2. Compute average F1 per vectorizer across all models → grouped bar chart
3. Find best single (model, vectorizer) combo by F1
4. Save decision: `outputs/results/best_vectorizer.json`
5. Write analysis: why TF-IDF wins (sparse linear classifiers, bigrams, sublinear_tf)

**Expected outcome:** TF-IDF with bigrams will be the winner. SVM + TF-IDF is the strongest baseline.

---

### Phase 5 — Notebook 05: Stacking

**Stacking design (uses best vectorizer from Phase 4):**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

base_estimators = [
    ('nb',   ComplementNB(alpha=0.1)),
    ('svm',  CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))),
    ('rf',   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('dt',   DecisionTreeClassifier(max_depth=20, random_state=42)),
    ('ada',  AdaBoostClassifier(n_estimators=100, random_state=42)),
]
meta_learner = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,           # out-of-fold predictions prevent data leakage
    stack_method='auto',  # uses predict_proba if available
    n_jobs=-1,
    passthrough=False
)
```

**`cv=5` is critical** — prevents data leakage into meta-features (see "Stacked Learners: The Right Way" from course slides).

**Output:**
- Metrics for Stacking model
- Comparison table: Stacking vs. best single model
- Confusion matrix saved to `outputs/confusion_matrices/stacking_final.png`
- 5 correct + 5 incorrect prediction examples

---

## 7. Research Findings (Key Facts for Implementation)

### 7.1 Existing Benchmark Results on This Dataset

From GitHub repos (parvvaresh/SnappFood, minisnappfood/SnappFood) that tackled this exact dataset:

| Model | Test Accuracy |
|---|---|
| LSTM | 84.39% |
| SVM (linear) | 81.00% |
| Random Forest | 80.65% |
| Logistic Regression | 80.60% |
| Decision Tree | 72.10% |
| KNN | 68.20% |
| GaussianNB | 67.90% |

State-of-the-art (ParsBERT transformer): ~88–90% F1 — not relevant for this project.

Our expected range with TF-IDF + classical ML: **68–82% accuracy**.

### 7.2 Why Hazm Normalization is Mandatory

Persian and Arabic share Unicode characters that look visually identical but have different code points:

| Character | Arabic | Persian |
|---|---|---|
| ye (ی) | U+064A (ي) | U+06CC (ی) |
| ke (ک) | U+0643 (ك) | U+06A9 (ک) |

Without normalization, the same word (e.g., کتاب) typed on Arabic vs. Persian keyboards becomes two different tokens. In a 70k user-review dataset, the same word appears in both encodings randomly throughout the data — inflating vocabulary, splitting TF-IDF weights, and making Word2Vec treat them as unrelated. Hazm's `Normalizer` collapses these to one canonical form. This is conceptually equivalent to `.lower()` in English.

### 7.3 Word2Vec: Why Train from Scratch

The project explicitly expects training from scratch (not using pretrained FastText etc.). On 70k reviews with domain-specific food/restaurant vocabulary, a corpus-trained model captures in-domain terms (غذا = food, سفارش = order, تحویل = delivery) better in relative terms. At this corpus size, `vector_size=100` is appropriate — 300d would be underfit.

### 7.4 MultinomialNB + Word2Vec Incompatibility

`MultinomialNB` requires non-negative feature values (it models word counts). Word2Vec vectors contain negative floats → will raise `ValueError`. Solution: use `GaussianNB` for Word2Vec experiments only (models continuous features with Gaussian distribution).

### 7.5 SVM + Stacking: The `predict_proba` Problem

`sklearn.svm.SVC(probability=False)` by default has no `predict_proba`. `StackingClassifier` needs probabilities. Two solutions:
- `SVC(probability=True)` — slow (uses Platt scaling internally)
- `CalibratedClassifierCV(LinearSVC(...))` — **preferred**: `LinearSVC` is much faster for large text, and wrapping it adds calibrated probabilities

### 7.6 Why TF-IDF Usually Wins on This Task

1. Downweights extremely common Persian words (even after stopword removal, some semi-stopwords remain)
2. `sublinear_tf=True` prevents one very-repeated word from dominating a review's vector
3. Bigrams (`ngram_range=(1,2)`) capture negation+adjective pairs like نه خوب (not good) or خیلی بد (very bad)
4. Sparse representation pairs perfectly with Linear SVM (which is the fastest and best-performing classifier in high-dimensional sparse spaces)

### 7.7 Stacking Theory (from Course Slides)

From `IML-Ensemble.md`:
- **First attempt (wrong):** train base models on D, use their predictions on same D to train meta-learner → base models overfit D, meta-features look nothing like test cases
- **Right way:** for each training point x, train base models on D-x (leave one out / cross-validation), then predict on x → meta-features look like test cases
- `StackingClassifier(cv=5)` implements the right way via 5-fold cross-validation
- Meta-learner is typically `LogisticRegression` (regularized, avoids meta-overfitting)

### 7.8 Persian NLP Tooling

| Library | Purpose | Install |
|---|---|---|
| `hazm` | Normalization, tokenization, lemmatization, stopwords | `pip install hazm` |
| `gensim` | Word2Vec training and loading | `pip install gensim` |
| `scikit-learn` | All ML models, vectorizers, metrics | `pip install scikit-learn` |
| `pandas` | Data loading and manipulation | `pip install pandas` |
| `numpy` | Numerical arrays | `pip install numpy` |
| `matplotlib` | Plotting | `pip install matplotlib` |
| `seaborn` | Heatmaps for confusion matrices | `pip install seaborn` |
| `scipy` | Sparse matrix save/load | `pip install scipy` |
| `joblib` | Saving models | included with sklearn |

Install all: `pip install hazm gensim scikit-learn pandas numpy matplotlib seaborn scipy`

---

## 8. GPU Note

Hardware: **NVIDIA RTX 4070 Laptop, CUDA 12.7**

PyTorch and TensorFlow show CUDA unavailable in Python (not installed), but `nvidia-smi` confirms the GPU is present and working.

**For this project, the GPU provides zero speedup** because:
- `scikit-learn` is CPU-only for all models (NB, SVM, RF, KNN, DT, AdaBoost, StackingClassifier)
- `gensim` Word2Vec is CPU-only
- 70k samples with classical ML completes in **under 20 minutes total on CPU**

If GPU acceleration were ever needed for future phases (e.g., LSTM, BERT), install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

---

## 9. Report Style Guide

**Tone (from `writing-style.txt`):**
- Completely conversational, like explaining to a friend — not formal
- Write as if thinking out loud; use "انگار", "خب", "یعنی", "درواقع", "بخاطر همین", "پس"
- Be honest about uncertainty: if something is not 100% clear, say so directly
- Technical terms (SVM, TF-IDF, Word2Vec, stacking, etc.) stay in English inside `\lr{}`
- No unnecessary bullet lists — prefer flowing paragraphs
- Comparisons structured as: "خوبیش چیه؟ ... بدیش چیه؟" self-Q&A format
- End each section with "پس" or "پس در نهایت" summary sentence

**LaTeX rules:**
- Numbers inside `\lr{}` or inside math `$ $`
- No math environments for plain text
- `itemize` only for genuine lists (not for paragraph-structured content)
- Font: Vazir (already configured in template)
- Dark theme already set in template (dark background, AccentBlue/AccentGreen headings)

**LaTeX template location:** `report/4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.tex`
- Title: `\projecttitle` needs to be filled with project name
- All student info already set in preamble commands

---

## 10. Actual Experiment Results (already run — do not re-run)

All 5 notebooks have been executed and outputs saved. Key findings:

### Phase 3 — Base Model F1 Scores

| Model | CountVectorizer | TF-IDF | Word2Vec |
|---|---|---|---|
| ComplementNB | 0.8358 | 0.8298 | — |
| GaussianNB | — | — | 0.7960 |
| SVM (Linear) | 0.8153 | 0.8313 | 0.8301 |
| DecisionTree | 0.7952 | 0.7946 | 0.7415 |
| RandomForest | **0.8384** | 0.8376 | 0.8249 |
| KNN | 0.7595 | 0.6782 | 0.8087 |
| AdaBoost | 0.7953 | 0.7961 | 0.8115 |

### Phase 4 — Best Vectorizer

Avg F1: CountVectorizer=0.8066 > Word2Vec=0.8021 > TF-IDF=0.7946

**Selected: CountVectorizer** (highest average F1 across all models)

> Note: CountVectorizer beat TF-IDF here — the opposite of our prior expectation.
> Likely reason: Hazm lemmatization already normalises term frequency differences,
> reducing the benefit of IDF weighting. CountVec bigrams alone sufficed.

### Phase 5 — Stacking Results

| | F1 | Accuracy |
|---|---|---|
| RandomForest (best single) | 0.8384 | 0.8386 |
| StackingClassifier | **0.8453** | **0.8455** |
| Δ | +0.0069 | +0.0069 |

**Stacking improves over the best single model by +0.69% F1.**

### Output Files (all saved)
- `data/processed/` — train.csv, test.csv, all 6 feature matrices, y_train/y_test
- `models/word2vec/persian_w2v.model` — gensim Word2Vec
- `outputs/confusion_matrices/` — 18 PNG files (all model×vectorizer combos)
- `outputs/results/phase3_results.csv` — full metrics table
- `outputs/results/best_vectorizer.json` — Phase 4 decision
- `outputs/results/phase5_stacking_results.json` — final comparison
- `outputs/figures/` — EDA, vectorizer norms, comparison charts

---

## 11. Session Continuation Protocol

When starting a new session:
1. Read this file (`PLAN.md`)
2. Check which notebooks exist and which are complete in `notebooks/`
3. Check `outputs/results/` for saved CSV files to know which phases are done
4. Check `data/processed/` for saved feature matrices
5. Continue from the first incomplete phase
6. Do NOT re-derive the plan — follow this document

**Implementation order (strict):**
```
src/ modules → Notebook 01 → Notebook 02 → Notebook 03 → Notebook 04 → Notebook 05 → LaTeX report
```

Each notebook reads outputs of the previous one. Never skip ahead.

---

## 11. Known Pitfalls (Do Not Repeat These)

- **Hazm stopwords include sentiment words:** خوب (good), عالی (excellent), بهتر (better) are in `hazm.stopwords_list()` by default. We build a custom stopword set by subtracting a `_SENTIMENT_PRESERVE` set. This is implemented in `src/preprocessing.py` — do not revert to raw `stopwords_list()`.


- `MultinomialNB` + Word2Vec → `ValueError` (negative values). Use `GaussianNB` for W2V.
- `SVC` without `probability=True` or `CalibratedClassifierCV` → `StackingClassifier` fails.
- `KNN` on raw 50k-dim TF-IDF sparse vectors → extremely slow. Use as-is but warn in notebook.
- Hazm `Lemmatizer` output for verbs: `"نوشت#نویس"` → always `.split('#')[0]`.
- gensim v4 API: access vectors via `model.wv['word']` not `model['word']`.
- `scipy.sparse.save_npz` / `load_npz` for sparse matrices; `np.save` / `np.load` for dense.
- Always read CSV with `encoding='utf-8'` — Persian text will corrupt otherwise.
- `StackingClassifier` with `n_jobs=-1` + `cv=5` is slow; expect 5–10 min on 56k train samples.
