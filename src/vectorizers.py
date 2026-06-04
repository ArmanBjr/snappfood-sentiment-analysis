"""
Custom sklearn-compatible vectorizer for Word2Vec document embeddings.

Why a custom transformer?
  gensim's Word2Vec model is not a sklearn transformer, so it cannot be used
  directly inside pipelines or the experiment loop. MeanWord2VecVectorizer
  wraps it to expose fit() / transform() so it behaves like CountVectorizer
  or TfidfVectorizer in the experiment code.

Document vector strategy: simple mean of word vectors.
  For each document, collect vectors of all tokens that exist in the W2V
  vocabulary, then average them. Tokens not in the vocabulary are skipped.
  Documents where no token is in-vocabulary get a zero vector.

  Simple mean works well for sentiment because emotional signal is spread
  across the whole review, not concentrated in a few rare words.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
