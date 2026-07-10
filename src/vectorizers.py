"""
Custom sklearn-compatible vectorizers for the experiment loop.

Why custom transformers?
  gensim's Word2Vec model is not a sklearn transformer, so it cannot be used
  directly inside pipelines or the experiment loop. The wrappers here expose
  fit() / transform() so a Word2Vec model behaves like CountVectorizer or
  TfidfVectorizer in the experiment code.

Three document-vector strategies live here:

  * MeanWord2VecVectorizer      — plain mean of in-vocab word vectors.
  * TfidfWeightedWord2Vec       — IDF-weighted mean. Rare, informative words
                                  (the ones that actually carry sentiment) get
                                  more weight than frequent filler words.
                                  Consistently beats plain mean pooling on
                                  short-review sentiment (Arora SIF, and the
                                  classic "weighted bag-of-vectors" result).

And one helper for the sparse side:

  * build_word_char_union()     — a FeatureUnion of a word-level TF-IDF and a
                                  character n-gram TF-IDF. Char n-grams are
                                  robust to the heavy spelling variation and
                                  informal morphology of Persian food-delivery
                                  reviews and reliably lift classical SVM by a
                                  few points over words alone.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class MeanWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Transform a list of pre-cleaned text strings into document vectors
    by averaging the Word2Vec vectors of all recognised tokens.

    Parameters
    ----------
    model       : trained gensim Word2Vec model  (model.wv gives KeyedVectors)
    vector_size : dimensionality of the word vectors (must match model)

    Usage
    -----
    >>> from gensim.models import Word2Vec
    >>> w2v = Word2Vec.load('models/word2vec/persian_w2v.model')
    >>> vec = MeanWord2VecVectorizer(model=w2v, vector_size=100)
    >>> X = vec.transform(X_train_clean)   # shape (n_samples, 100)
    """

    def __init__(self, model, vector_size: int = 100):
        self.model = model
        self.vector_size = vector_size

    def fit(self, X, y=None):
        # Nothing to fit — the Word2Vec model is already trained externally.
        return self

    def transform(self, X) -> np.ndarray:
        """
        Parameters
        ----------
        X : iterable of strings (pre-cleaned, space-separated tokens)

        Returns
        -------
        np.ndarray of shape (len(X), vector_size)
        """
        result = np.zeros((len(X), self.vector_size), dtype=np.float32)

        for i, text in enumerate(X):
            tokens = text.split() if isinstance(text, str) else []
            # Collect vectors only for tokens that exist in the W2V vocabulary
            vecs = [
                self.model.wv[token]
                for token in tokens
                if token in self.model.wv
            ]
            if vecs:
                result[i] = np.mean(vecs, axis=0)
            # else: row stays as zero vector (OOV document)

        return result

    def get_feature_names_out(self):
        # Satisfies the sklearn transformer interface for pipelines
        return np.array([f'w2v_dim_{i}' for i in range(self.vector_size)])


class TfidfWeightedWord2Vec(BaseEstimator, TransformerMixin):
    """
    Transform pre-cleaned text into document vectors by an IDF-weighted mean
    of the Word2Vec vectors of in-vocabulary tokens.

    Each token's contribution is scaled by its inverse-document-frequency, so
    a rare sentiment word (افتضاح, عالی) dominates the document vector while a
    high-frequency filler word barely moves it. This is the cheap, strong
    cousin of Arora's SIF and beats plain mean pooling on short reviews.

    Unlike MeanWord2VecVectorizer, this transformer HAS state: it must learn
    the IDF table from the training corpus in fit(), so always fit on TRAIN
    only and transform train/test separately to avoid leakage.

    Parameters
    ----------
    model        : trained gensim Word2Vec model (model.wv gives KeyedVectors)
    vector_size  : dimensionality of the word vectors (must match model)
    default_idf  : 'max' gives OOV-but-known tokens the largest seen IDF;
                   any float overrides with a fixed fallback weight.

    Usage
    -----
    >>> vec = TfidfWeightedWord2Vec(model=w2v, vector_size=100)
    >>> Xtr = vec.fit_transform(X_train_clean)   # learns IDF from train
    >>> Xte = vec.transform(X_test_clean)        # reuses train IDF
    """

    def __init__(self, model, vector_size: int = 100, default_idf='max'):
        self.model = model
        self.vector_size = vector_size
        self.default_idf = default_idf

    def fit(self, X, y=None):
        # Learn IDF over the same tokenization the documents already use
        # (whitespace split — text arrives pre-tokenized from preprocessing).
        tfidf = TfidfVectorizer(
            analyzer=str.split,
            lowercase=False,
            token_pattern=None,
        )
        tfidf.fit(X if isinstance(X, (list, tuple)) else list(X))
        self.idf_ = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        self._fallback_idf_ = (
            max(tfidf.idf_) if self.default_idf == 'max' else float(self.default_idf)
        )
        return self

    def transform(self, X) -> np.ndarray:
        result = np.zeros((len(X), self.vector_size), dtype=np.float32)
        wv = self.model.wv

        for i, text in enumerate(X):
            tokens = text.split() if isinstance(text, str) else []
            acc = np.zeros(self.vector_size, dtype=np.float32)
            weight_sum = 0.0
            for token in tokens:
                if token not in wv:
                    continue
                w = self.idf_.get(token, self._fallback_idf_)
                acc += w * wv[token]
                weight_sum += w
            if weight_sum > 0.0:
                result[i] = acc / weight_sum
            # else: row stays zero (no in-vocab tokens)

        return result

    def get_feature_names_out(self):
        return np.array([f'w2v_idf_dim_{i}' for i in range(self.vector_size)])


def build_word_char_union(
    *,
    word_ngram_range=(1, 2),
    char_ngram_range=(3, 5),
    word_max_features: int = 50000,
    char_max_features: int = 50000,
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """
    Build a FeatureUnion that concatenates a word-level TF-IDF and a
    character n-gram TF-IDF into one sparse feature matrix.

    Character n-grams ('char_wb' = word-boundary-aware) catch informal Persian
    spelling, elongation, attached negation morphemes, and typos that the
    word vectorizer misses entirely. Stacking word + char features is a
    well-worn, low-risk way to lift a classical SVM a few points on noisy,
    short, user-generated text — and it keeps the required word-level
    representation intact as one half of the union.

    The result plugs into the experiment loop exactly like any other
    vectorizer: it exposes fit / transform / fit_transform and produces a
    scipy sparse matrix.

    Returns
    -------
    sklearn.pipeline.FeatureUnion
    """
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=word_ngram_range,
        max_features=word_max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=char_ngram_range,
        max_features=char_max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    return FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf),
    ])


def oov_rate(texts, model) -> float:
    """
    Compute the fraction of unique tokens in `texts` that are
    out-of-vocabulary for the given Word2Vec model.

    Useful diagnostic to print in notebooks after training W2V.
    """
    vocab = set(model.wv.key_to_index.keys())
    all_tokens = set(token for text in texts for token in text.split())
    if not all_tokens:
        return 0.0
    oov = all_tokens - vocab
    return len(oov) / len(all_tokens)
