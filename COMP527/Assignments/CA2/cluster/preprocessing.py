"""
preprocessing.py
────────────────
Loads a dataset file in the format:

    ID<TAB>Sentence

Cleans the sentences, then runs the following pipeline on ALL 1,760 rows
(duplicates intentionally retained):

    embed (all-mpnet-base-v2, 768-dim)
    -> L2 normalise
    -> Adaptive PCA  (data-driven 80 % variance threshold)
    -> UMAP          (cosine metric, n_components=5, min_dist=0.0,
                      n_neighbors=10)
    -> features.npy  (N x 5)

Why duplicates are retained
---------------------------
This dataset contains repeated Notre Dame / Wikipedia sentences (18 unique
sentences appearing 4-5 times each, totalling 80 rows).  These repetitions
are treated as a structural density feature rather than noise: in UMAP's
k-nearest-neighbour graph, the repeated sentences reinforce each other's
neighbourhood connections, producing a denser, more isolated manifold region
that AHC can detect as a distinct cluster.  Removing duplicates reduces the
Notre Dame sub-corpus to 18 points (1.1 % of the unique data), which is
insufficient to form a stable geometric cluster at n_neighbors=10.

UMAP n_neighbors=10 (reduced from the default 15)
--------------------------------------------------
A smaller n_neighbours makes UMAP more sensitive to local, dense structure.
With n_neighbors=10 and 80 Notre Dame rows, each Notre Dame sentence finds
most of its 10 nearest neighbours within the Notre Dame group itself,
producing a tighter, more isolated cluster in the 5-dimensional output space.
The BERTopic documentation notes that decreasing n_neighbors "creates more
local structure", which is the intended effect here.

Outputs
-------
embeddings.npy  -- (N, 768) float32   raw all-mpnet-base-v2 embeddings
sentences.pkl   -- list of N cleaned sentences (all rows, including dups)
features.npy    -- (N, 5)   float32   UMAP feature matrix for clustering

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
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

EMBEDDINGS_OUT = "embeddings.npy"
SENTENCES_OUT  = "sentences.pkl"
FEATURES_OUT   = "features.npy"

BATCH_SIZE   = 64
RANDOM_STATE = 42

# Adaptive PCA: retain minimum components explaining this fraction of variance.
PCA_VARIANCE_THRESHOLD = 0.80

# UMAP parameters.
#
# metric='cosine'   : correct for L2-normalised text embeddings; builds the
#                     kNN neighbourhood graph using cosine distance, consistent
#                     with the MNR training objective of all-mpnet-base-v2.
#
# n_components=5    : standard output dimensionality for clustering tasks
#                     (BERTopic standard; Grootendorst, 2022).
#
# min_dist=0.0      : recommended for clustering by the UMAP documentation --
#                     packs points within the same neighbourhood as tightly as
#                     the manifold permits.  Mathematically valid: min_dist
#                     simply sets the lower bound on the optimised pairwise
#                     distances in the low-dimensional layout.
#
# n_neighbors=10    : reduced from the default 15 to make UMAP more sensitive
#                     to small, dense local structures (e.g., the 80-row Notre
#                     Dame sub-corpus).  Each Notre Dame sentence finds most of
#                     its 10 nearest neighbours within the Notre Dame group,
#                     producing a tighter, more isolated cluster region.
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS  = 10
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """
    Read the tab-separated file, correctly handling sentences that span
    multiple physical lines due to embedded newline characters.
    Ensures all 1,760 instances are loaded and each receives a label.
    """
    records: list = []
    current_id: int | None = None
    current_sentence: list = []

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            if current_id is not None:
                records.append({
                    "ID": current_id,
                    "Sentence": " ".join(current_sentence).strip()
                })
            current_id = int(parts[0].strip())
            current_sentence = [parts[1].rstrip("\n")]
        else:
            if current_id is not None:
                current_sentence.append(line.rstrip("\n"))

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
    Heavy normalisation is intentionally excluded: all-mpnet-base-v2 is
    trained on natural, cased, punctuated English text.
    """
    if not isinstance(text, str):
        return "empty"
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+",   " ", text)
    text = text.strip()
    return text if text else "empty"


def preprocess(df: pd.DataFrame) -> List[str]:
    """Apply clean_text to every row, preserving exact row count and order."""
    sentences = df["Sentence"].apply(clean_text).tolist()
    print(f"[clean] {len(sentences):,} sentences after cleaning")
    return sentences


def embed(sentences: List[str]) -> np.ndarray:
    """
    Encode all sentences (including duplicates) with all-mpnet-base-v2.
    Identical sentences receive identical embedding vectors; their repeated
    presence in the feature matrix provides density that helps UMAP and AHC
    identify them as a compact cluster.
    normalize_embeddings=False: applied explicitly in the next step.
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
    L2-normalise each row so all vectors lie on the unit hypersphere.
    Makes Euclidean distance monotonically equivalent to cosine distance,
    ensuring Ward-linkage AHC is consistent with the MNR training objective.
    """
    print("\n[norm] Applying L2 normalisation ...")
    X_norm = normalize(X)
    print(f"[norm] Shape after normalisation: {X_norm.shape}")
    return X_norm.astype(np.float32)


def apply_pca(X: np.ndarray) -> np.ndarray:
    """
    Adaptive PCA: retain the minimum number of components whose cumulative
    explained variance >= PCA_VARIANCE_THRESHOLD.
    Denoises the embedding matrix and reduces UMAP's kNN graph cost.
    """
    print(f"\n[pca] Fitting full PCA to compute adaptive component count ...")
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X)

    cumvar       = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD)) + 1
    n_components = max(10, min(n_components, min(X.shape) - 1))

    print(f"[pca] Adaptive selection: {n_components} components "
          f"(threshold={PCA_VARIANCE_THRESHOLD:.0%})")

    pca   = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[pca] {X.shape} -> {X_pca.shape}  "
          f"(cumulative variance retained: {explained:.4f})")
    return X_pca.astype(np.float32)


def apply_umap(X_pca: np.ndarray) -> np.ndarray:
    """
    UMAP non-linear manifold projection.

    n_neighbors=10 (reduced from default 15):
        Makes UMAP more sensitive to small, dense local structures.  With
        n_neighbors=10, each of the 80 Notre Dame rows finds most of its
        10 nearest neighbours within the Notre Dame group itself, producing
        a more isolated manifold region that AHC can detect as a distinct
        cluster.

    metric='cosine':
        Correct for L2-normalised text embeddings.

    min_dist=0.0:
        Recommended by the UMAP documentation for clustering tasks.
        Mathematically valid: it sets the lower bound on optimised pairwise
        distances in the low-dimensional layout, producing compact clusters
        without introducing any artificial structure.
    """
    try:
        from umap import UMAP
    except ImportError:
        print("\n[umap] ERROR: umap-learn is not installed.")
        print("       Run:  pip install umap-learn")
        sys.exit(1)

    print(f"\n[umap] Applying UMAP: ({X_pca.shape[0]}, {X_pca.shape[1]}) "
          f"-> ({X_pca.shape[0]}, {UMAP_N_COMPONENTS})")
    print(f"[umap] n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric='{UMAP_METRIC}', "
          f"random_state={RANDOM_STATE}")

    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=False,
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"[umap] UMAP complete. Output shape: {X_umap.shape}")
    return X_umap.astype(np.float32)


def preprocess_pipeline(data_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Full preprocessing pipeline:

        load -> clean -> embed (all rows) -> L2 normalise
        -> adaptive PCA -> UMAP

    Duplicates are retained throughout so that repeated sentences provide
    density in the UMAP neighbourhood graph.

    Returns
    -------
    X_umap    : np.ndarray, shape (N, 5)
        UMAP feature matrix for clustering.
    sentences : List[str], length N
        Cleaned sentences aligned row-for-row with X_umap.
    embeddings : np.ndarray, shape (N, 768)
        Raw all-mpnet-base-v2 embeddings.
    """
    df         = load_data(data_path)
    sentences  = preprocess(df)
    embeddings = embed(sentences)
    X_norm     = l2_normalise(embeddings)
    X_pca      = apply_pca(X_norm)
    X_umap     = apply_umap(X_pca)
    return X_umap, sentences, embeddings


def save_artifacts(
    sentences: List[str],
    embeddings: np.ndarray,
    features: np.ndarray,
) -> None:
    """Persist raw embeddings, cleaned sentences, and UMAP feature matrix."""
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
    print("\n[done] Preprocessing complete.")


if __name__ == "__main__":
    main()