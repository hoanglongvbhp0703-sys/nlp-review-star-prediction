import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clean_data import clean_dataframe, clean_text
from src.tokenize_data import load_vectorizer

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "review.csv")
TEXT_COL = "Text"
LABEL_COL = "Score"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lstm_rating_model.keras")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "artifacts", "vectorizer.pkl")
NUM_CLASSES = 5


def normalize_labels(y: np.ndarray) -> np.ndarray:
    y = y.astype("int32")
    if y.min() == 1 and y.max() > 1:
        return y - 1
    return y


def split_like_training(df):
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    n_train = int(0.9 * len(df))
    n_val = int(0.05 * len(df))
    test_df = df.iloc[n_train + n_val :].copy()
    return test_df


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = load_vectorizer(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def evaluate_test_set():
    model, vectorizer = load_assets()
    df = clean_dataframe(CSV_PATH, text_col=TEXT_COL)
    test_df = split_like_training(df)
    x_test = vectorizer(test_df[TEXT_COL].astype(str).values).numpy().astype("int32")
    y_true = normalize_labels(test_df[LABEL_COL].values.astype("int32"))
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype("int32")
    return y_true, y_pred


def draw_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ticks = np.arange(NUM_CLASSES)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)
    ax.set_xlabel("Predicted star")
    ax.set_ylabel("True star")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            value = cm[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    fig.tight_layout()
    return fig


def draw_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    true_counts = np.bincount(y_true + 1, minlength=NUM_CLASSES + 1)[1:]
    pred_counts = np.bincount(y_pred + 1, minlength=NUM_CLASSES + 1)[1:]
    x = np.arange(1, NUM_CLASSES + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, true_counts, width=width, label="True")
    ax.bar(x + width / 2, pred_counts, width=width, label="Pred")
    ax.set_xticks(x)
    ax.set_xlabel("Star")
    ax.set_ylabel("Count")
    ax.set_title("Star Distribution (Test Set)")
    ax.legend()
    fig.tight_layout()
    return fig


st.set_page_config(page_title="NLP Star Predictor", layout="wide")
st.title("NLP Review Rating Dashboard")
st.caption("Enter a comment to predict star rating and view distribution plus the model confusion matrix.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("Missing model/vectorizer. Please train first: `python train_model/train_lstm.py`.")
    st.stop()

model, vectorizer = load_assets()

st.subheader("1) Predict comment rating")
comment = st.text_area("Enter comment", height=120, placeholder="Example: the product is fairly good for the price")
if st.button("Predict"):
    if not comment.strip():
        st.warning("You have not entered a comment.")
    else:
        cleaned = clean_text(comment)
        x = vectorizer(np.array([cleaned])).numpy().astype("int32")
        prob = model.predict(x, verbose=0)[0]
        pred_star = int(np.argmax(prob) + 1)
        st.success(f"Prediction: {pred_star} stars")
        st.bar_chart({f"{i + 1} stars": float(prob[i]) for i in range(NUM_CLASSES)})

st.subheader("2) Star distribution and confusion matrix")
y_true, y_pred = evaluate_test_set()

col1, col2 = st.columns(2)
with col1:
    st.pyplot(draw_distribution(y_true, y_pred), use_container_width=True)
with col2:
    st.pyplot(draw_confusion_matrix(y_true, y_pred), use_container_width=True)
