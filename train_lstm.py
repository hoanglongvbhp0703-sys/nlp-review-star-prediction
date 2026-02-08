import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from clean_data import clean_dataframe
from tokenize_data import build_vectorizer, save_vectorizer
from padding_data import make_padded_inputs
from lstm_model import build_lstm_classifier


def make_tf_dataset(
    x_ids: np.ndarray,
    y: np.ndarray,
    batch_size=32,
    shuffle=True,
):
    ds = tf.data.Dataset.from_tensor_slices((x_ids, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x_ids), 10000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def normalize_labels(y: np.ndarray) -> np.ndarray:
    if y is None:
        return y
    y = y.astype("int32")
    if y.min() == 1 and y.max() > 1:
        return y - 1
    return y


if __name__ == "__main__":
    CSV_PATH = r"review.csv"
    TEXT_COL = "Text"
    LABEL_COL = "Score"

    df = clean_dataframe(CSV_PATH, text_col=TEXT_COL)

    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    n_train = int(0.9 * len(df))
    n_val = int(0.05 * len(df))
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    vectorizer = build_vectorizer(
        train_df[TEXT_COL].astype(str).values,
        vocab_size=10000,
        max_len=200,
    )
    save_vectorizer(vectorizer, "vectorizer.pkl")
    vocab_size = len(vectorizer.get_vocabulary())
    max_len = vectorizer.get_config().get("output_sequence_length", 200)

    x_train, y_train, _ = make_padded_inputs(
        train_df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        vectorizer=vectorizer,
    )
    x_val, y_val, _ = make_padded_inputs(
        val_df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        vectorizer=vectorizer,
    )
    x_test, y_test, _ = make_padded_inputs(
        test_df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        vectorizer=vectorizer,
    )
    y_train = normalize_labels(y_train)
    y_val = normalize_labels(y_val)
    y_test = normalize_labels(y_test)
    train_ds = make_tf_dataset(x_train, y_train, batch_size=64, shuffle=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=64, shuffle=False)
    test_ds = make_tf_dataset(x_test, y_test, batch_size=64, shuffle=False)

    model = build_lstm_classifier(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=64,
        num_classes=5,
        lstm_units=128,
        lstm_dropout=0.25,
        l2_strength=0.002,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks
    )
    model.save("lstm_rating_model.keras")
    print("Model saved to lstm_rating_model.keras")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # Plot loss/accuracy curves
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("loss", []), label="loss")
    plt.plot(epochs, hist.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("accuracy", []), label="accuracy")
    plt.plot(epochs, hist.get("val_accuracy", []), label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot test metrics
    plt.figure(figsize=(4, 3))
    plt.bar(["test_loss", "test_acc"], [test_loss, test_acc], color=["#E45756", "#4C78A8"])
    plt.title("Test Metrics")
    plt.tight_layout()
    plt.show()