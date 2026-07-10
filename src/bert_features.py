"""
ParsBERT feature extraction for the bonus / high-accuracy phase.

Why this exists
---------------
The required pipeline (CountVectorizer / TF-IDF / Word2Vec + classical models +
stacking) tops out around the bag-of-words ceiling for this dataset. A
contextual transformer breaks that ceiling because it understands word ORDER
and CONTEXT — exactly what a bag of words throws away. On the Snappfood
sentiment task, ParsBERT-based features land several points above the classical
ceiling.

This module gives you the *cheap half* of that win: frozen feature extraction.
We run ParsBERT once as a fixed encoder, mean-pool its last hidden states into
one 768-dim vector per review, and hand those vectors to the SAME classical
models / stacking ensemble already in the project. No gradient updates, no GPU
required (CPU works, just slower). It slots into the experiment loop like any
other dense vectorizer.

Full fine-tuning (unfreezing ParsBERT and training end-to-end) squeezes out the
last couple of points but needs a GPU and a training loop; that lives in the
bonus notebook, not here.

Model
-----
HooshvareLab/bert-fa-base-uncased — ParsBERT v2, the standard Persian BERT in
modern checkpoint format. First call downloads ~600 MB from the HuggingFace hub
and caches it locally. (The older v1 id randomizes LayerNorm on transformers>=5;
see DEFAULT_MODEL_NAME below.)

Design notes
------------
  * MEAN pooling over real tokens (mask-aware) beats raw [CLS] for frozen
    feature extraction on sentence classification — [CLS] is only well-shaped
    after fine-tuning.
  * Feed the ORIGINAL Persian text here, NOT the cleaned/negation-tagged tokens.
    ParsBERT has its own WordPiece tokenizer and was pretrained on natural text;
    our EMO_/NEG_ markers and aggressive cleaning would only confuse it.
  * Batched, torch.no_grad(), eval mode — this is inference only.

Dependencies (install in the bonus notebook, not required for the core project):
    pip install torch transformers
"""

from __future__ import annotations

import numpy as np

# ParsBERT v2 (modern checkpoint format). The older v1 id
# 'HooshvareLab/bert-base-parsbert-uncased' stores LayerNorm params under the
# legacy 'gamma'/'beta' names; transformers >=5 no longer auto-renames them, so
# loading v1 silently random-initializes all 25 LayerNorm layers (missing_keys =
# 'bert.*.LayerNorm.weight/bias'). Always use the v2 id below.
DEFAULT_MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'


class ParsBertFeatureExtractor:
    """
    Frozen ParsBERT encoder that turns raw Persian strings into 768-dim
    mean-pooled sentence embeddings.

    Lazy-loads torch/transformers on first use so importing this module never
    forces the heavy dependencies on the core (classical) pipeline.

    Parameters
    ----------
    model_name : HuggingFace model id.
    max_length : WordPiece truncation length. Snappfood reviews are short;
                 128 covers virtually all of them without wasting compute.
    batch_size : how many reviews to encode per forward pass.
    device     : 'cuda', 'cpu', or None to auto-detect.
    pooling    : 'mean' (mask-aware average, recommended) or 'cls'.

    Usage
    -----
    >>> enc = ParsBertFeatureExtractor(batch_size=32)
    >>> X_train = enc.transform(df_train['comment'].tolist())   # (n, 768)
    >>> X_test  = enc.transform(df_test['comment'].tolist())
    >>> np.save('data/processed/X_train_bert.npy', X_train)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
        batch_size: int = 32,
        device: str | None = None,
        pooling: str = 'mean',
    ):
        if pooling not in ('mean', 'cls'):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self._device = device
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                'ParsBertFeatureExtractor needs torch + transformers. '
                'Install them in the bonus notebook with: '
                'pip install torch transformers'
            ) from exc

        self._torch = torch
        if self._device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model, loading_info = AutoModel.from_pretrained(
            self.model_name, output_loading_info=True
        )
        # Guard against silent legacy-format failures: if any encoder weight is
        # missing it was random-initialized (e.g. old gamma/beta LayerNorm names
        # dropped by transformers>=5), which poisons every extracted feature.
        bert_missing = [
            k
            for k in loading_info.get('missing_keys', [])
            if k.startswith(('bert.', 'encoder.', 'embeddings.'))
        ]
        if bert_missing:
            raise RuntimeError(
                f'{self.model_name} loaded with {len(bert_missing)} randomly '
                f'initialized encoder weights (e.g. {bert_missing[:3]}). This '
                'checkpoint is likely a legacy gamma/beta format incompatible '
                'with this transformers version. Use a modern checkpoint such '
                "as 'HooshvareLab/bert-fa-base-uncased'."
            )
        self._model.to(self._device)
        self._model.eval()

    def fit(self, X=None, y=None) -> 'ParsBertFeatureExtractor':
        # Frozen encoder — nothing to learn. Present so it behaves like an
        # sklearn transformer inside pipelines.
        self._ensure_loaded()
        return self

    def transform(self, texts) -> np.ndarray:
        """
        Encode an iterable of raw Persian strings into mean-pooled embeddings.

        Returns
        -------
        np.ndarray of shape (len(texts), hidden_size)   # 768 for base ParsBERT
        """
        self._ensure_loaded()
        torch = self._torch

        texts = [
            t if isinstance(t, str) and t.strip() else '[UNK]'
            for t in texts
        ]

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self._device)

            with torch.no_grad():
                out = self._model(**enc)

            hidden = out.last_hidden_state  # (B, T, H)
            if self.pooling == 'cls':
                pooled = hidden[:, 0, :]
            else:
                mask = enc['attention_mask'].unsqueeze(-1).type_as(hidden)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            all_vecs.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)

    def fit_transform(self, texts, y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    def get_feature_names_out(self):
        return np.array([f'bert_dim_{i}' for i in range(768)])


def extract_and_save(
    df_train,
    df_test,
    text_col: str = 'comment',
    out_dir: str = 'data/processed',
    **extractor_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: encode train and test text with frozen ParsBERT and save the
    feature matrices to disk so downstream notebooks can load them without
    re-running the encoder.

    Writes:
        {out_dir}/X_train_bert.npy
        {out_dir}/X_test_bert.npy

    Returns the (X_train, X_test) arrays as well.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    enc = ParsBertFeatureExtractor(**extractor_kwargs)

    X_train = enc.transform(df_train[text_col].tolist())
    X_test = enc.transform(df_test[text_col].tolist())

    np.save(os.path.join(out_dir, 'X_train_bert.npy'), X_train)
    np.save(os.path.join(out_dir, 'X_test_bert.npy'), X_test)

    return X_train, X_test
