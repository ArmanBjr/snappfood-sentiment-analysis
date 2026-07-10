"""
Persian text preprocessing pipeline for the Snappfood sentiment dataset.

This version is tuned for SENTIMENT accuracy, not generic text cleaning.
The default configuration deliberately keeps sentiment-bearing signal that a
naive cleaning pipeline throws away:

  * Negation is TAGGED, not deleted  (Pang & Lee 2002: prefix NEG_ to the words
    after a negator). Negation is the single most important sentiment feature.
  * Emojis are MAPPED to EMO_POS / EMO_NEG tokens, not stripped — in food-delivery
    reviews emojis carry loud polarity.
  * Character elongation can be kept as an EMPH_ marker (خیلیییی = emphatic "very").
  * Stopword removal is OFF by default. For TF-IDF/Count the IDF term already
    down-weights frequent words, and Hazm's stoplist contains negators and
    intensifiers that flip sentiment. Turn it on only for an ablation.

Every step is switchable via clean_text(...) keyword args so the notebook can run
ablations (stopwords on/off, lemma on/off, negation on/off, emoji-map on/off).

Pipeline order (default):
  1. Remove URLs
  2. Map sentiment emojis  -> EMO_POS / EMO_NEG  (before symbols are stripped)
  3. Hazm normalization    (Arabic->Persian chars, ZWNJ, diacritics)
  4. Remove remaining emojis / non-Persian symbols
  5. Remove digits
  6. Remove punctuation (kept as soft sentence boundaries for negation scope)
  7. Collapse char elongation (optionally emit EMPH_ marker)
  8. Tokenize (Hazm WordTokenizer)
  9. Negation tagging (NEG_ prefix within a window, reset on conjunctions)
 10. [optional] Remove stopwords (sentiment words always preserved)
 11. [optional] Lemmatize (special EMO_/NEG_/EMPH_ tokens are left untouched)
 12. Drop tokens shorter than 2 characters (special tokens kept)
 13. Re-join into a single space-separated string
"""

import re
import pandas as pd
from hazm import Normalizer, WordTokenizer, Lemmatizer, stopwords_list


# --------------------------------------------------------------------------- #
#  Singleton objects — instantiate once so every call shares the same object  #
# --------------------------------------------------------------------------- #
_normalizer = Normalizer()
_tokenizer = WordTokenizer()
_lemmatizer = Lemmatizer()

# Hazm's default stopwords include sentiment-critical words. Even when stopword
# removal is enabled we never drop these.
_SENTIMENT_PRESERVE = {
    'خوب', 'بد', 'عالی', 'افتضاح', 'بهتر', 'بدتر', 'ضعیف', 'قوی',
    'لذیذ', 'مزخرف', 'راضی', 'ناراضی', 'خوش', 'ناخوش', 'دوست',
    'متاسف', 'خوشمزه', 'بدمزه', 'تازه', 'کثیف', 'تمیز', 'گرم', 'سرد',
    'دیر', 'زود', 'خراب', 'سالم', 'عالیه', 'بدبود', 'نمیکنم',
}
_stopwords = set(stopwords_list()) - _SENTIMENT_PRESERVE

# Persian negators. Standalone tokens after which sentiment flips.
# (Verb-internal negation like نمی‌خواهم is partly handled by Hazm tokenization;
#  these catch the separable / colloquial negators that survive tokenization.)
_NEGATORS = {
    'نه', 'نَه', 'نیست', 'نیستش', 'نبود', 'نبودش', 'نداره', 'ندارد', 'نداشت',
    'نشد', 'نمیشه', 'نمی‌شه', 'نکرد', 'نکن', 'نخور', 'بدون', 'هیچ',
    'نخواهد', 'نباید', 'نمیکنم', 'نمیکنه', 'نمیشد', 'بی', 'نا',
}
# Conjunctions / boundaries that END a negation scope.
_NEG_RESET = {'اما', 'ولی', 'ولیکن', 'اگرچه', 'هرچند', 'با_اینکه', 'فقط'}

# Sentiment emojis -> mapped tokens. Stripped of variation selectors first.
_POS_EMOJI = set('😋😍👌👍🙏❤😊🤤💯🔥😁😄🥰😘🤩☺🙂😻👏✨💕💚💙😇🤗')
_NEG_EMOJI = set('🤮🤬👎😡🤢💩😠😤😣😞😢😭�", 😖😩🙄😒💔👎🤡😕☹')
# remove the stray accidental chars if any slipped in:
_NEG_EMOJI = {c for c in _NEG_EMOJI if c not in {'"', ',', ' '}}

# Compiled regex patterns
_RE_URL = re.compile(r'https?://\S+|www\.\S+|t\.me/\S+|@\w+')
_RE_VARIATION = re.compile(r'[︀-️\U0001F3FB-\U0001F3FF‍]')  # selectors + skin tones + ZWJ
_RE_EMOJI = re.compile(
    r'[\U0001F000-\U0001FAFF'   # supplementary symbol/pictograph planes
    r'\U00002600-\U000027BF'    # misc symbols + dingbats
    r'\U0001F1E0-\U0001F1FF'    # flags
    r'←-⇿'            # arrows
    r'⬀-⯿'            # misc symbols & arrows
    r']',
    flags=re.UNICODE,
)
_RE_DIGITS = re.compile(r'[0-9۰-۹٠-٩]')
_RE_PUNCT = re.compile(r'[!؟،؛.,:;()\[\]{}\-_"\'«»…/\\|@#$%^&*+=<>~`]')
_RE_REPEAT = re.compile(r'(.)\1{2,}')        # 3+ consecutive identical chars
_RE_ELONG = re.compile(r'(.)\1{2,}')         # used to detect elongation for EMPH_
_RE_WHITESPACE = re.compile(r'\s+')

# Sentinel tokens that must never be lemmatized / stopword-filtered / length-dropped.
_SPECIAL_PREFIXES = ('EMO_', 'NEG_', 'EMPH_')


def _map_emojis(text: str) -> str:
    """Replace sentiment emojis with EMO_POS / EMO_NEG marker tokens."""
    out = []
    for ch in text:
        if ch in _POS_EMOJI:
            out.append(' EMO_POS ')
        elif ch in _NEG_EMOJI:
            out.append(' EMO_NEG ')
        else:
            out.append(ch)
    return ''.join(out)


def _apply_negation(tokens, window: int = 3):
    """
    Prefix tokens that fall inside a negation scope with 'NEG_'.

    Scope starts right after a negator and lasts `window` tokens or until a
    reset conjunction (اما / ولی ...). Heuristic, in the spirit of Pang & Lee.
    """
    out, neg = [], 0
    for t in tokens:
        if t in _NEG_RESET:
            neg = 0
            out.append(t)
            continue
        if t in _NEGATORS:
            out.append(t)
            neg = window
            continue
        if neg > 0 and not t.startswith(_SPECIAL_PREFIXES):
            out.append('NEG_' + t)
            neg -= 1
        else:
            out.append(t)
    return out


def clean_text(
    text: str,
    *,
    remove_stopwords: bool = False,
    handle_negation: bool = True,
    map_emoji: bool = True,
    lemmatize: bool = True,
    mark_elongation: bool = False,
    neg_window: int = 3,
    min_token_len: int = 2,
) -> str:
    """
    Run the preprocessing pipeline on a single Persian string.

    Parameters
    ----------
    remove_stopwords : drop Hazm stopwords (sentiment words always kept). Default OFF.
    handle_negation  : tag NEG_ after negators (Pang & Lee). Default ON.
    map_emoji        : convert sentiment emojis to EMO_POS / EMO_NEG. Default ON.
    lemmatize        : Hazm lemmatization (special tokens untouched). Default ON.
    mark_elongation  : emit EMPH_<token> for elongated words. Default OFF.
    neg_window       : how many tokens a negation scope covers.
    min_token_len    : drop ordinary tokens shorter than this (special tokens kept).

    Returns a space-joined string of clean tokens (possibly empty).
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    # 1. URLs / handles
    text = _RE_URL.sub(' ', text)

    # 2. Map sentiment emojis BEFORE symbol stripping
    if map_emoji:
        text = _map_emojis(text)

    # 3. Normalize (Arabic->Persian, half-space, diacritics)
    text = _normalizer.normalize(text)

    # 4. Remove leftover (neutral) emojis and symbols, plus variation selectors
    text = _RE_VARIATION.sub('', text)
    text = _RE_EMOJI.sub(' ', text)

    # 5. Digits
    text = _RE_DIGITS.sub('', text)

    # 6. Punctuation
    text = _RE_PUNCT.sub(' ', text)

    # 7. Elongation: optionally keep an emphasis marker, then collapse
    if mark_elongation:
        def _emph(m):
            return f' EMPH_{m.group(1)} '
        # Replace the elongated run itself with a marker token (this both
        # collapses the run and records that emphasis happened).
        text = _RE_ELONG.sub(_emph, text)
    else:
        text = _RE_REPEAT.sub(r'\1', text)

    text = _RE_WHITESPACE.sub(' ', text).strip()
    if not text:
        return ''

    # 8. Tokenize
    tokens = _tokenizer.tokenize(text)

    # 9. Negation tagging
    if handle_negation:
        tokens = _apply_negation(tokens, window=neg_window)

    # 10. Stopwords (optional). Never drop special or sentiment tokens.
    if remove_stopwords:
        tokens = [
            t for t in tokens
            if t.startswith(_SPECIAL_PREFIXES) or t not in _stopwords
        ]

    # 11. Lemmatize ordinary tokens only. NEG_ must be checked BEFORE the
    # general _SPECIAL_PREFIXES check (NEG_ is itself one of those prefixes),
    # otherwise every negated token would fall into the "leave untouched"
    # branch and never get its underlying word lemmatized.
    if lemmatize:
        lem = []
        for t in tokens:
            if t.startswith('NEG_'):
                lem.append('NEG_' + _lemmatizer.lemmatize(t[4:]).split('#')[0])
            elif t.startswith(_SPECIAL_PREFIXES):
                lem.append(t)
            else:
                lem.append(_lemmatizer.lemmatize(t).split('#')[0])
        tokens = lem

    # 12. Drop very short ordinary tokens (keep special tokens)
    tokens = [
        t for t in tokens
        if t.startswith(_SPECIAL_PREFIXES) or len(t) >= min_token_len
    ]

    return ' '.join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = 'comment',
    label_col: str = 'label',
    **clean_kwargs,
) -> pd.DataFrame:
    """
    Apply clean_text() to every row and return a copy with an added 'cleaned'
    column. The ORIGINAL text column is preserved (needed later for BERT and for
    human-readable prediction examples in the report).

    Extra keyword args are forwarded to clean_text (e.g. remove_stopwords=True)
    so the notebook can run ablations without editing this file.
    """
    result = df[[text_col, label_col]].copy()
    result['cleaned'] = result[text_col].apply(lambda t: clean_text(t, **clean_kwargs))

    empty_count = (result['cleaned'].str.strip() == '').sum()
    if empty_count > 0:
        print(f"[preprocessing] Dropped {empty_count} rows with empty text after cleaning.")
        result = result[result['cleaned'].str.strip() != ''].reset_index(drop=True)

    return result


def show_examples(df_raw: pd.DataFrame, df_clean: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a side-by-side DataFrame of n before/after examples for notebooks."""
    idx = df_raw.index[:n]
    return pd.DataFrame({
        'Before': df_raw.loc[idx, 'comment'].values,
        'After': df_clean.loc[idx, 'cleaned'].values,
        'Label': df_raw.loc[idx, 'label'].values,
    })
