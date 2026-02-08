import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.clean_data import clean_dataframe
from src.tokenize_data import build_vectorizer, save_vectorizer
from src.padding_data import make_padded_inputs
from train_model.lstm_model import build_lstm_classifier


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


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    return float(np.mean(f1))


if __name__ == "__main__":
    CSV_PATH = os.path.join(ROOT, "data", "review.csv")
    TEXT_COL = "Text"
    LABEL_COL = "Score"

    ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
    MODELS_DIR = os.path.join(ROOT, "models")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
    MODEL_PATH = os.path.join(MODELS_DIR, "lstm_rating_model.keras")

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
    save_vectorizer(vectorizer, VECTORIZER_PATH)
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
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype("int32")
    test_f1 = compute_macro_f1(y_true, y_pred, num_classes=5)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # Additional evaluation metrics
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=5).numpy()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    y_true_stars = y_true + 1
    y_pred_stars = y_pred + 1
    within_1_star = np.mean(np.abs(y_true_stars - y_pred_stars) <= 1)
    mae_stars = np.mean(np.abs(y_true_stars - y_pred_stars))
    print(f"% predictions within <= 1 star: {within_1_star * 100:.2f}%")
    print(f"MAE on star labels (1..5): {mae_stars:.4f}")

    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)
    last_epoch = epochs[-1]

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
