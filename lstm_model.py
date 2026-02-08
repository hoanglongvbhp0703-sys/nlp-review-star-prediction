import tensorflow as tf

from embedding_data import build_embedding_layer


def build_lstm_classifier(
    vocab_size: int,
    max_len: int,
    embedding_dim: int,
    lstm_units: int = 64,
    num_classes: int = 5,
    lstm_dropout: float = 0.25,
    l2_strength: float = 0.002,
):
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="token_ids")
    x = build_embedding_layer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        mask_zero=True,
    )(inputs)

    # Embedding -> BiLSTM -> Dense(softmax)
    kernel_reg = tf.keras.regularizers.l2(l2_strength) if l2_strength > 0 else None
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm_units,
            activation="tanh",
            dropout=lstm_dropout,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            kernel_regularizer=kernel_reg,
            recurrent_regularizer=kernel_reg,
        )
    )(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=kernel_reg,
        name="rating",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
