import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
from typing import Tuple


def build_vectorizer(
    texts,
    vocab_size=10000,
    max_len=200,
)->tf.keras.layers.TextVectorization:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None,
        split="whitespace",
    )
    vectorizer.adapt(texts)
    return vectorizer


def vectorize_texts(
    texts,
    vectorizer: tf.keras.layers.TextVectorization,
) -> np.ndarray:
    return vectorizer(texts).numpy().astype(np.int32)


def vectorize_dataframe(
    df: pd.DataFrame,
    text_col: str,
    vectorizer: tf.keras.layers.TextVectorization,
) -> np.ndarray:
    return vectorize_texts(df[text_col].astype(str).values, vectorizer)


def save_vectorizer(vectorizer, path="vectorizer.pkl"):
    config = vectorizer.get_config()
    vocabulary = vectorizer.get_vocabulary()
    with open(path, "wb") as f:
        pickle.dump({"config": config, "vocabulary": vocabulary}, f)


def load_vectorizer(path="vectorizer.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    vectorizer = tf.keras.layers.TextVectorization.from_config(obj["config"])
    if "vocabulary" not in obj:
        raise ValueError("No vocabulary found in vectorizer.pkl")
    vectorizer.set_vocabulary(obj["vocabulary"])
    return vectorizer
