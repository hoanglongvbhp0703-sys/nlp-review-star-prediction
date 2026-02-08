import numpy as np
import tensorflow as tf


def build_embedding_layer(
    embedding_dim: int = 128,
    vocab_size: int | None = None,
    vectorizer: tf.keras.layers.TextVectorization | None = None,
    mask_zero: bool = True,
    trainable: bool = True,
) -> tf.keras.layers.Embedding:
    if vocab_size is None:
        if vectorizer is None:
            raise ValueError("Provide vocab_size or vectorizer.")
        vocab_size = vectorizer.vocabulary_size()
    emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=mask_zero,
        trainable=trainable,
        name="embedding",
    )
    return emb



def embed_padded_ids(
    padded_ids: np.ndarray,
    embedding_layer: tf.keras.layers.Embedding,
    batch_size: int = 256,
) -> np.ndarray:
    # For inference/feature extraction only. Not for training.
    ds = tf.data.Dataset.from_tensor_slices(padded_ids).batch(batch_size)
    outs = []
    for x in ds:
        outs.append(embedding_layer(x))
    out = tf.concat(outs, axis=0)
    return out.numpy()


