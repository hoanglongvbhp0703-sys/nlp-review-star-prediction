import numpy as np
import pandas as pd
import tensorflow as tf

from tokenize_data import load_vectorizer, build_vectorizer


def _vectorizer_matches(
    vectorizer: tf.keras.layers.TextVectorization,
    vocab_size: int,
    max_len: int,
) -> bool:
    cfg = vectorizer.get_config()
    return (
        cfg.get("max_tokens") == vocab_size
        and cfg.get("output_sequence_length") == max_len
    )


def build_or_load_vectorizer(
    texts,
    vocab_size=10000,
    max_len=200,
    vectorizer_path="vectorizer.pkl",
    rebuild=False,
) -> tf.keras.layers.TextVectorization:
    if not rebuild:
        try:
            vec = load_vectorizer(vectorizer_path)
            if _vectorizer_matches(vec, vocab_size, max_len):
                return vec
        except Exception:
            pass

    vec = build_vectorizer(texts, vocab_size=vocab_size, max_len=max_len)
    return vec


def make_padded_inputs(
    df: pd.DataFrame,
    text_col: str,
    label_col: str | None = None,
    vectorizer: tf.keras.layers.TextVectorization | None = None,
    vocab_size: int = 10000,
    max_len: int = 200,
    vectorizer_path: str = "vectorizer.pkl",
    rebuild_vectorizer: bool = False,
):
    texts = df[text_col].astype(str).values
    if vectorizer is None:
        vectorizer = build_or_load_vectorizer(
            texts=texts,
            vocab_size=vocab_size,
            max_len=max_len,
            vectorizer_path=vectorizer_path,
            rebuild=rebuild_vectorizer,
        )

    x_ids = vectorizer(texts).numpy().astype(np.int32)
    y = None
    if label_col is not None and label_col in df.columns:
        y = df[label_col].values.astype("int32")
    return x_ids, y, vectorizer
