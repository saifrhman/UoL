"""
preprocessing.py
────────────────
Loads a dataset file in the format:

    ID<TAB>Sentence

Cleans the sentences, produces sentence embeddings using:
    sentence-transformers/all-MiniLM-L12-v2

Then applies:
    1. L2 normalisation  – ensures all embedding vectors are unit vectors,
                           so subsequent PCA decomposes directional (semantic)
                           variance rather than magnitude variance.
    2. PCA (100 components) – reduces dimensionality while retaining ~74.6 % of
                           variance. The resulting components are kept as-is
                           (not re-normalised): PCA deliberately assigns more
                           variance to earlier components, and Euclidean
                           distance in this space naturally weights the most
                           informative components more heavily. Re-normalising
                           after PCA would erase this useful structure and
                           degrade cluster separation.

Outputs
-------
embeddings.npy  – (N, 384) float32  raw sentence embeddings
sentences.pkl   – cleaned sentence list aligned row-for-row with embeddings
features.npy    – (N, 100)  float32  PCA-reduced feature matrix for clustering

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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDINGS_OUT = "embeddings.npy"
SENTENCES_OUT  = "sentences.pkl"
FEATURES_OUT   = "features.npy"

BATCH_SIZE     = 64
RANDOM_STATE   = 42

# 100 PCA components retain ~74.6 % of variance from the 384-dim MiniLM space.
# Retaining 100 components is necessary to preserve the fine-grained geometric
# separation of the small Wikipedia sub-corpus from the larger modern English
# cluster — a signal lost when fewer components are used.
PCA_COMPONENTS = 100
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """
    Read the tab-separated file, correctly handling sentences that span
    multiple physical lines due to embedded newline characters.
    Lines starting with an integer ID followed by a tab begin a new record;
    all other lines are continuations of the previous sentence.
    This ensures all 1,760 instances are loaded and receive a label.
    """
    records = []
    current_id = None
    current_sentence = []

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            # New record — save the previous one first
            if current_id is not None:
                records.append({
                    "ID": current_id,
                    "Sentence": " ".join(current_sentence).strip()
                })
            current_id = int(parts[0].strip())
            current_sentence = [parts[1].rstrip("\n")]
        else:
            # Continuation line — append to current sentence
            if current_id is not None:
                current_sentence.append(line.rstrip("\n"))

    # Save the final record
    if current_id is not None:
        records.append({
            "ID": current_id,
            "Sentence": " ".join(current_sentence).strip()
        })

    df = pd.DataFrame(records)

    required_cols = {"ID", "Sentence"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Input file must contain columns {required_cols}, "
            f"but got {set(df.columns)}"
        )
    print(f"[load] {len(df):,} rows loaded from '{path}'")
    return df

def clean_text(text: str) -> str:
    """
    Light-touch cleaning that preserves linguistic content.

    Steps applied:
      - Remove escaped quotation marks
      - Collapse repeated whitespace / tabs into a single space
      - Collapse actual newline characters into a single space.
      - Strip leading / trailing whitespace

    Heavy normalisation (lowercasing, punctuation removal, stopword removal)
    is intentionally omitted: MiniLM is a transformer model that benefits
    from natural, cased, punctuated sentence text.
    """
    if not isinstance(text, str):
        return "empty"
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+",    " ", text)
    text = text.strip()
    return text if text else "empty"


def preprocess(df: pd.DataFrame) -> List[str]:
    """
    Apply clean_text to every row while preserving exact row count and order.
    Order preservation is critical: label.txt must align 1-to-1 with the
    original dataset rows.
    """
    sentences = df["Sentence"].apply(clean_text).tolist()
    print(f"[clean] {len(sentences):,} sentences after cleaning")
    return sentences


def embed(sentences: List[str]) -> np.ndarray:
    """
    Encode sentences with MiniLM and return raw (un-normalised) embeddings.

    normalize_embeddings=False because normalisation is applied explicitly in
    the next step, keeping pipeline stages distinct and transparent.
    """
    print(f"[embed] Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] Encoding {len(sentences):,} sentences "
          f"(batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    print(f"[embed] Embedding matrix shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def l2_normalise(X: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row so that all embedding vectors are unit vectors.

    Applied once — before PCA — so that PCA decomposes angular (semantic)
    variance between sentences rather than differences in vector magnitude.
    PCA output is deliberately left un-normalised: its variance weighting
    (more variance in earlier components) is informative signal that helps
    Euclidean-based AHC distinguish fine-grained semantic clusters.
    """
    print("\n[norm] Applying L2 normalisation ...")
    X_norm = normalize(X)
    print(f"[norm] Shape after normalisation: {X_norm.shape}")
    return X_norm.astype(np.float32)


def apply_pca(X: np.ndarray) -> np.ndarray:
    """
    Reduce dimensionality from 384 to PCA_COMPONENTS with PCA.

    PCA is fit on the L2-normalised embeddings, ensuring it captures
    directional (semantic) differences between sentences.  The output is
    NOT re-normalised: the variance-weighted geometry of PCA space helps
    Ward-linkage AHC weight informative dimensions more heavily, which
    improves cluster separation compared to a uniform (re-normalised) space.
    """
    print(f"\n[pca] Reducing dimensions from {X.shape[1]} to {PCA_COMPONENTS} ...")
    reducer  = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca    = reducer.fit_transform(X)
    explained = reducer.explained_variance_ratio_.sum()
    print(f"[pca] {X.shape} -> {X_pca.shape}")
    print(f"[pca] Cumulative explained variance retained: {explained:.4f}")
    return X_pca.astype(np.float32)


def preprocess_pipeline(data_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Full preprocessing pipeline:

        load  →  clean  →  embed  →  L2 normalise  →  PCA

    Returns
    -------
    X_pca : np.ndarray, shape (N, 100)
        PCA-reduced feature matrix used for AHC clustering and silhouette
        evaluation.  Variance-weighted: earlier components carry more
        semantic signal and naturally receive more weight in Euclidean distance.
    sentences : List[str]
        Cleaned sentences aligned row-for-row with X_pca.
    embeddings : np.ndarray, shape (N, 384)
        Raw MiniLM embeddings before any normalisation or reduction.
    """
    df         = load_data(data_path)
    sentences  = preprocess(df)
    embeddings = embed(sentences)
    X          = l2_normalise(embeddings)
    X_pca      = apply_pca(X)
    return X_pca, sentences, embeddings


def save_artifacts(
    sentences: List[str],
    embeddings: np.ndarray,
    features: np.ndarray,
) -> None:
    """Persist embeddings, cleaned sentences, and PCA-reduced features."""
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
    X, sentences, embeddings = preprocess_pipeline(data_path)
    save_artifacts(sentences, embeddings, X)
    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()