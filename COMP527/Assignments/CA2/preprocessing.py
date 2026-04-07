"""
preprocessing.py
────────────────
Loads a dataset file in the format:

    ID<TAB>Sentence

Cleans the sentences, produces sentence embeddings using:
    sentence-transformers/all-MiniLM-L12-v2

Then applies:
    1. L2 normalization
    2. PCA for initial dimensionality reduction
    3. UMAP for non-linear dimensionality reduction

Outputs
-------
embeddings.npy   – (N, 384) float32 numpy array
sentences.pkl    – cleaned sentence list aligned with the embeddings
features.npy     – final transformed feature matrix after normalization + PCA + UMAP

Usage
-----
python preprocessing.py data_train.txt
"""

from __future__ import annotations

import re
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:
    print("Error: umap-learn is required. Install it with:")
    print("pip install umap-learn")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDINGS_OUT = "embeddings.npy"
SENTENCES_OUT = "sentences.pkl"
FEATURES_OUT = "features.npy"

BATCH_SIZE = 64
RANDOM_STATE = 42

PCA_COMPONENTS = 50
UMAP_COMPONENTS = 16
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """Read the tab-separated file and return a DataFrame with ID + Sentence."""
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df.columns = df.columns.str.strip()

    required_cols = {"ID", "Sentence"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Input file must contain columns {required_cols}, but got {set(df.columns)}"
        )

    print(f"[load] {len(df):,} rows loaded from '{path}'")
    return df


def clean_text(text: str) -> str:
    """
    Light-touch cleaning that preserves linguistic content:
      - unescape literal \\n sequences
      - remove escaped quotes
      - collapse whitespace
      - strip leading/trailing whitespace

    Heavy normalization such as lowercasing, punctuation removal, or stopword
    removal is intentionally skipped because MiniLM works better with natural
    sentence text.
    """
    if not isinstance(text, str):
        return "empty"

    # Unescape escaped newlines from the raw file
    text = text.replace("\\n", " ")

    # Remove escaped quotes that appear in the dataset
    text = text.replace('\\"', '"').replace("\\'", "'")

    # Collapse tabs/spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)

    text = text.strip()

    # Preserve row count exactly
    if len(text) == 0:
        return "empty"

    return text


def preprocess(df: pd.DataFrame) -> List[str]:
    """
    Apply cleaning while preserving row count and original order exactly.
    This is important because label.txt must align with the original dataset.
    """
    sentences = df["Sentence"].apply(clean_text).tolist()
    print(f"[clean] {len(sentences):,} sentences after cleaning")
    return sentences


def embed(sentences: List[str]) -> np.ndarray:
    """Generate sentence embeddings in batches."""
    print(f"[embed] Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    print(
        f"[embed] Encoding {len(sentences):,} sentences "
        f"(batch_size={BATCH_SIZE}) ..."
    )

    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalization at embedding stage
        convert_to_numpy=True,
    )

    print(f"[embed] Embedding matrix shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to the feature matrix.
    This is useful for cosine-based similarity and clustering.
    """
    print("\n[norm] Applying L2 normalization ...")
    X_norm = normalize(X)
    print(f"[norm] Feature matrix shape after normalization: {X_norm.shape}")
    return X_norm.astype(np.float32)


def apply_pca(X: np.ndarray) -> np.ndarray:
    """
    Apply PCA as an initial dimensionality reduction step to mitigate the
    curse of dimensionality while preserving most of the variance.
    """
    print(f"\n[pca] Reducing dimensions to {PCA_COMPONENTS} ...")
    reducer = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_reduced = reducer.fit_transform(X)
    explained = reducer.explained_variance_ratio_.sum()

    print(f"[pca] {X.shape} -> {X_reduced.shape}")
    print(f"[pca] Total explained variance retained: {explained:.4f}")
    return X_reduced.astype(np.float32)


def apply_umap(X: np.ndarray) -> np.ndarray:
    """
    Apply UMAP after PCA to capture non-linear structures and preserve local
    neighbourhood relationships.
    """
    print(f"\n[umap] Reducing dimensions to {UMAP_COMPONENTS} ...")
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    X_reduced = reducer.fit_transform(X)
    print(f"[umap] {X.shape} -> {X_reduced.shape}")
    return X_reduced.astype(np.float32)


def preprocess_pipeline(data_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Full preprocessing pipeline:
        load data -> clean text -> embed -> normalize -> PCA -> UMAP

    Returns
    -------
    features : np.ndarray
        Final transformed feature matrix used for clustering.
    sentences : List[str]
        Cleaned sentences aligned with features.
    embeddings : np.ndarray
        Raw sentence embeddings before PCA/UMAP.
    """
    df = load_data(data_path)
    sentences = preprocess(df)
    embeddings = embed(sentences)
    X = normalize_features(embeddings)
    X = apply_pca(X)
    X = apply_umap(X)
    return X, sentences, embeddings


def save_artifacts(
    sentences: List[str],
    embeddings: np.ndarray,
    features: np.ndarray
) -> None:
    """Save embeddings, cleaned sentences, and final transformed features."""
    np.save(EMBEDDINGS_OUT, embeddings)
    print(f"[save] Embeddings saved -> '{EMBEDDINGS_OUT}'")

    with open(SENTENCES_OUT, "wb") as f:
        pickle.dump(sentences, f)
    print(f"[save] Sentences saved  -> '{SENTENCES_OUT}'")

    np.save(FEATURES_OUT, features)
    print(f"[save] Features saved   -> '{FEATURES_OUT}'")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python preprocessing.py data_train.txt")
        sys.exit(1)

    data_path = sys.argv[1]

    if not Path(data_path).exists():
        print(f"Error: file not found -> {data_path}")
        sys.exit(1)

    features, sentences, embeddings = preprocess_pipeline(data_path)
    save_artifacts(sentences, embeddings, features)

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()