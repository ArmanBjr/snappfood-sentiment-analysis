"""
Persian text preprocessing pipeline for the Snappfood sentiment dataset.

Preprocessing order:
  1. Hazm normalization  (Arabic→Persian chars, ZWNJ, diacritics)
  2. Remove URLs
  3. Remove emojis / non-Persian symbols
  4. Remove digits (Western + Arabic-Indic + Extended Arabic-Indic)
  5. Remove punctuation
  6. Collapse repeated characters  (خوووب → خوب)
  7. Tokenize  (Hazm WordTokenizer)
  8. Remove stopwords  (Hazm stopwords_list)
  9. Lemmatize  (Hazm Lemmatizer, take past-stem before '#')
  10. Drop tokens shorter than 2 characters
  11. Re-join into a single space-separated string
"""

import re
import pandas as pd
from hazm import Normalizer, WordTokenizer, Lemmatizer, stopwords_list


# --------------------------------------------------------------------------- #
#  Singleton objects — instantiate once so every call shares the same object  #
# --------------------------------------------------------------------------- #
_normalizer = Normalizer()
_tokenizer  = WordTokenizer()
_lemmatizer = Lemmatizer()

# Hazm's default stopwords include sentiment-critical words (خوب, عالی, بهتر).
# We build a custom set that removes those so the models can actually learn from them.
_SENTIMENT_PRESERVE = {
    'خوب', 'بد', 'عالی', 'افتضاح', 'بهتر', 'بدتر', 'ضعیف', 'قوی',
    'لذیذ', 'مزخرف', 'راضی', 'ناراضی', 'خوش', 'ناخوش', 'دوست',
    'متاسف', 'خوشمزه', 'بدمزه', 'تازه', 'کثیف', 'تمیز',
}
_stopwords = set(stopwords_list()) - _SENTIMENT_PRESERVE

# Compiled regex patterns
_RE_URL       = re.compile(r'https?://\S+|www\.\S+')
_RE_EMOJI     = re.compile(
    r'[\U00010000-\U0010FFFF'   # supplementary planes (most emojis)
    r'\U0001F600-\U0001F64F'    # emoticons
    r'\U0001F300-\U0001F5FF'    # misc symbols & pictographs
    r'\U0001F680-\U0001F6FF'    # transport & map
    r'\U0001F1E0-\U0001F1FF'    # flags
    r'☀-⛿'            # misc symbols
    r'✀-➿'            # dingbats
    r']',
    flags=re.UNICODE
)
_RE_DIGITS    = re.compile(r'[0-9۰-۹٠-٩]')
_RE_PUNCT     = re.compile(r'[!؟،؛.,:;()\[\]{}\-_"\'«»…\/\\|@#$%^&*+=<>~`]')
_RE_REPEAT    = re.compile(r'(.)\1{2,}')   # 3+ consecutive identical chars
_RE_WHITESPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """
    Run the full preprocessing pipeline on a single Persian string.

    Returns a space-joined string of clean lemmatized tokens,
    or an empty string if nothing survives the pipeline.
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    # 1. Remove URLs before normalization — normalizer strips ':' and '/'
    #    which breaks URL regex if applied after
    text = _RE_URL.sub(' ', text)

    # 2. Normalize: Arabic→Persian chars, half-space fix, diacritics removal
    text = _normalizer.normalize(text)

    # 3. Remove emojis and non-textual symbols
    text = _RE_EMOJI.sub(' ', text)

    # 4. Remove digits
    text = _RE_DIGITS.sub('', text)

    # 5. Remove punctuation
    text = _RE_PUNCT.sub(' ', text)

    # 6. Collapse repeated characters to 1 (خوووووب → خوب, عالیییی → عالی)
    #    Using \1 (not \1\1) so the result is always a valid Persian word root
    text = _RE_REPEAT.sub(r'\1', text)

    # 7. Collapse extra whitespace
    text = _RE_WHITESPACE.sub(' ', text).strip()

    if not text:
        return ''

    # 8. Tokenize
    tokens = _tokenizer.tokenize(text)

    # 9. Remove stopwords
    tokens = [t for t in tokens if t not in _stopwords]

    # 10. Lemmatize — output for verbs: "past_stem#present_stem"; take past_stem
    tokens = [_lemmatizer.lemmatize(t).split('#')[0] for t in tokens]

    # 11. Drop tokens shorter than 2 characters
    tokens = [t for t in tokens if len(t) >= 2]

    return ' '.join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = 'comment',
    label_col: str = 'label',
) -> pd.DataFrame:
    """
    Apply clean_text() to every row of a DataFrame and return a copy
    with an added 'cleaned' column.

    Parameters
    ----------
    df        : input DataFrame containing at least text_col
    text_col  : name of the column with raw Persian text
    label_col : name of the label column (kept as-is)

    Returns
    -------
    DataFrame with columns [text_col, label_col, 'cleaned']
    Rows where 'cleaned' is empty after preprocessing are dropped.
    """
    result = df[[text_col, label_col]].copy()
    result['cleaned'] = result[text_col].apply(clean_text)

    empty_count = (result['cleaned'] == '').sum()
    if empty_count > 0:
        print(f"[preprocessing] Dropped {empty_count} rows with empty text after cleaning.")
        result = result[result['cleaned'] != ''].reset_index(drop=True)

    return result


def show_examples(df_raw: pd.DataFrame, df_clean: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return a side-by-side DataFrame of n before/after examples.
    Useful for display in notebooks.
    """
    idx = df_raw.index[:n]
    return pd.DataFrame({
        'Before': df_raw.loc[idx, 'comment'].values,
        'After':  df_clean.loc[idx, 'cleaned'].values,
        'Label':  df_raw.loc[idx, 'label'].values,
    })
