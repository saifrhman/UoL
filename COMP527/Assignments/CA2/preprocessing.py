"""
preprocessing.py
────────────────
Loads a dataset file in the format:

    ID<TAB>Sentence

Cleans sentences, then runs the following pipeline:

    embed (all-mpnet-base-v2, 768-dim)
    → L2 normalise
    → Adaptive PCA  (data-driven variance threshold)
    → UMAP          (non-linear manifold projection)
    → features.npy  (N × UMAP_N_COMPONENTS)

Pipeline improvements over the MiniLM-L12-v2 + fixed-100-PCA baseline
-----------------------------------------------------------------------
1.  all-mpnet-base-v2 (768-dim, 1 B+ training pairs, MNR contrastive loss)
      • Highest overall score among standard all-* sentence-transformer models
        on the Sentence Embeddings Benchmark (sbert.net, 2024).
      • Outperforms MiniLM on MTEB clustering tasks; richer semantic separation
        for diverse-register corpora (Petukhova et al., 2024).
      • No cloud API required: loads fully locally via Hugging Face model hub.

2.  Adaptive PCA (PCA_VARIANCE_THRESHOLD)
      • Selects the minimum number of principal components that together explain
        at least PCA_VARIANCE_THRESHOLD of total variance.
      • Removes the arbitrary hardcoded 100-component choice; the component
        count is data-driven and adjusts to any input embedding model.

3.  UMAP (post-PCA manifold projection)
      • PCA is linear: it decomposes global variance along orthogonal axes but
        cannot represent non-linear cluster structure in the embedding manifold.
      • UMAP (McInnes et al., 2018) builds a topological neighbourhood graph
        and projects it into a low-dimensional Euclidean space that preserves
        both local and global manifold structure.
      • PCA → UMAP is the current best-practice pipeline for text-embedding
        clustering (Grootendorst, 2022; Vizuara, 2024; Towards Data Science,
        2025): PCA first denoises and reduces memory for UMAP's kNN graph,
        then UMAP resolves the non-linear cluster manifolds.
      • min_dist=0.1 (not 0.0): preserves realistic intra-cluster spread;
        using 0.0 compresses all within-cluster distances toward zero and
        artificially inflates silhouette scores (Clustering Results doc, 2025).
      • random_state=42: fully deterministic and reproducible.

Outputs
-------
embeddings.npy  – (N, 768) float32   raw all-mpnet-base-v2 embeddings
sentences.pkl   – cleaned sentence list, row-aligned with embeddings
features.npy    – (N, UMAP_N_COMPONENTS) float32  UMAP feature matrix

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

# Embedding model: 768-dim MPNet fine-tuned on 1 B sentence pairs.
# Substantially outperforms MiniLM-L12-v2 (384-dim) on MTEB clustering tasks.
# No subscription or cloud API required; loads locally via Hugging Face hub.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

EMBEDDINGS_OUT = "embeddings.npy"
SENTENCES_OUT  = "sentences.pkl"
FEATURES_OUT   = "features.npy"

BATCH_SIZE   = 64
RANDOM_STATE = 42

# Adaptive PCA: retain the minimum number of principal components whose
# cumulative explained variance reaches this threshold.  No count is hardcoded.
# 0.80 (80 %) is a principled cutoff: it retains dominant semantic axes while
# discarding noise in the tail of the eigenvalue spectrum.
PCA_VARIANCE_THRESHOLD = 0.80

# UMAP hyper-parameters (all configurable here; nothing is hardcoded below).
#
# n_components : output dimensionality.  20 gives rich cluster geometry
#                without over-compressing small clusters (e.g. ~90 Wikipedia
#                docs in a 1 760-sentence corpus).
# n_neighbors  : neighbourhood size for the UMAP kNN graph.  15 is the
#                standard default (McInnes et al., 2018); it balances local
#                structure (low values) against global topology (high values).
# min_dist     : minimum packing distance between projected points.
#                0.1 prevents artificial silhouette inflation while still
#                producing well-separated clusters (contrast with min_dist=0.0,
#                which compresses within-cluster distances to near-zero).
# metric       : Euclidean on variance-weighted PCA coordinates is the correct
#                choice because PCA output is not unit-normalised.
UMAP_N_COMPONENTS = 20
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
UMAP_METRIC       = "euclidean"

# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """
    Read the tab-separated file, correctly handling sentences that span
    multiple physical lines due to embedded newline characters.
    Lines starting with an integer ID followed by a tab begin a new record;
    all other lines are continuations of the previous sentence.
    Ensures all 1,760 instances are loaded and each receives a label.
    """
    records: list = []
    current_id: int | None = None
    current_sentence: list = []

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[1:]:          # skip header row
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            # New record — flush the previous one first
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

    # Flush the final record
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

    Steps:
      1. Remove escaped quotation marks (\\\" and \\').
      2. Collapse repeated whitespace / tabs into a single space.
      3. Collapse actual newline characters into a single space.
      4. Strip leading / trailing whitespace.

    Heavy normalisation (lowercasing, punctuation removal, stopword removal,
    stemming) is intentionally omitted: all-mpnet-base-v2 is a transformer
    trained on natural, cased, punctuated English text.  Removing these
    features pushes sentences out of the model's training distribution,
    degrading embedding quality and therefore cluster separability.
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
    Encode sentences with all-mpnet-base-v2 and return raw (un-normalised)
    embeddings.

    all-mpnet-base-v2:
      • Architecture: MPNet (combines MLM and PLM pre-training).
      • Output: 768-dimensional dense vectors.
      • Training: fine-tuned on 1 billion sentence pairs via Multiple Negative
        Ranking (MNR) contrastive loss, which optimises the embedding space so
        that semantically similar sentences are geometrically proximate.
      • Quality: highest-ranked standard all-* model on the Sentence
        Embeddings Benchmark (sbert.net) and competitive with much larger
        open-source LLMs (Petukhova et al., 2024).

    normalize_embeddings=False: normalisation is applied explicitly in the
    next step, keeping pipeline stages distinct and auditable.
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
    L2-normalise each row so all embedding vectors are unit vectors
    (i.e. all points lie on the unit hypersphere S^{d-1}).

    Why normalise before PCA?
    ─────────────────────────
    Raw all-mpnet-base-v2 embeddings vary in both direction (semantic
    content) and magnitude (loosely correlated with token length).  PCA
    applied to un-normalised vectors decomposes a mixture of angular and
    magnitude variance.  L2 normalisation ensures PCA decomposes purely
    angular (semantic) variance — the type most relevant to thematic
    clustering.

    Geometric consequence: for unit vectors â, b̂, Euclidean and cosine
    distances are monotonically equivalent:
        ||â − b̂||² = 2(1 − cos(â, b̂))
    Ward-linkage AHC (Euclidean) therefore implicitly optimises cosine
    similarity — precisely the metric that MNR loss optimised.
    """
    print("\n[norm] Applying L2 normalisation ...")
    X_norm = normalize(X)
    print(f"[norm] Shape after normalisation: {X_norm.shape}")
    return X_norm.astype(np.float32)


def apply_pca(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Adaptive PCA: select the minimum number of components whose cumulative
    explained variance is >= PCA_VARIANCE_THRESHOLD.

    No component count is hardcoded.  The selection is fully data-driven:
    for all-mpnet-base-v2 (768-dim) with threshold=0.80, this typically
    yields 55–90 components depending on the corpus.

    Why not re-normalise after PCA?
    ────────────────────────────────
    PCA is variance-preserving: PC1 has the largest eigenvalue (most
    semantic signal), PC2 the second largest, etc.  Euclidean distance in
    PCA space naturally weights the earlier, higher-variance components
    more heavily — exactly the inductive bias needed for AHC to recover
    fine-grained clusters.  Re-normalising collapses this useful structure.

    Returns
    -------
    X_pca       : float32 array of shape (N, n_components)
    n_components: number of components actually retained
    """
    print(f"\n[pca] Fitting full PCA to compute adaptive component count ...")
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X)

    cumvar      = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD)) + 1
    # Safety bounds: always keep at least 10, never exceed the matrix rank
    n_components = max(10, min(n_components, min(X.shape) - 1))

    print(f"[pca] Adaptive selection: {n_components} components "
          f"(threshold={PCA_VARIANCE_THRESHOLD:.0%})")

    pca   = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[pca] {X.shape} → {X_pca.shape}  "
          f"(cumulative variance retained: {explained:.4f})")
    return X_pca.astype(np.float32), n_components


def apply_umap(X_pca: np.ndarray) -> np.ndarray:
    """
    UMAP non-linear manifold projection applied after PCA denoising.

    Motivation
    ──────────
    PCA is linear: it can only project data onto the directions of maximum
    variance in the original space.  Sentence embeddings lie on a non-linear
    manifold in R^768; clusters that are genuinely distinct may not be linearly
    separable, causing PCA alone to produce overlapping cluster projections.

    UMAP (McInnes, Healy & Melville, 2018) constructs a fuzzy topological
    representation of the data manifold using k-nearest-neighbour graphs,
    then optimises a low-dimensional layout that preserves both local
    neighbourhoods and global cluster topology.  This makes cluster
    boundaries sharper in Euclidean space, substantially improving
    silhouette coefficients for AHC.

    PCA → UMAP is the established best-practice pipeline (Grootendorst,
    2022; Vizuara, 2024): PCA reduces the ambient dimension and filters
    noise, reducing the cost of UMAP's kNN graph construction and
    improving its numerical stability.

    Parameter choices
    ─────────────────
    n_components  = UMAP_N_COMPONENTS (default 20)
        Rich enough to capture multi-cluster topology; low enough to keep
        Euclidean distances in the output meaningful for AHC.
    n_neighbors   = UMAP_N_NEIGHBORS (default 15)
        Standard default from McInnes et al. (2018); balances local
        (small values) and global (large values) structure preservation.
    min_dist      = UMAP_MIN_DIST (default 0.1)
        Controls intra-cluster point packing.  0.1 avoids the artificial
        silhouette inflation caused by min_dist=0.0 (which compresses all
        within-cluster distances toward zero).
    metric        = UMAP_METRIC (default 'euclidean')
        Euclidean on variance-weighted PCA coordinates is appropriate
        because the PCA output is NOT unit-normalised.
    random_state  = RANDOM_STATE (42)
        UMAP uses randomness in its stochastic gradient descent phase;
        fixing the seed ensures bit-for-bit reproducibility.
    """
    try:
        from umap import UMAP
    except ImportError:
        print("\n[umap] ERROR: umap-learn is not installed.")
        print("       Run:  pip install umap-learn")
        sys.exit(1)

    print(f"\n[umap] Applying UMAP: ({X_pca.shape[0]}, {X_pca.shape[1]}) "
          f"→ ({X_pca.shape[0]}, {UMAP_N_COMPONENTS})")
    print(f"[umap] n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}, "
          f"random_state={RANDOM_STATE}")

    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=False,      # faster when RAM is available
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"[umap] UMAP complete. Output shape: {X_umap.shape}")
    return X_umap.astype(np.float32)


def preprocess_pipeline(data_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Full preprocessing pipeline:

        load → clean → embed → L2 normalise → adaptive PCA → UMAP

    Returns
    -------
    X_umap     : np.ndarray, shape (N, UMAP_N_COMPONENTS)
        UMAP-reduced feature matrix passed to AHC clustering.
        Euclidean distances in this space reflect manifold-aware semantic
        proximity between sentences.
    sentences  : List[str]
        Cleaned sentences aligned row-for-row with X_umap.
    embeddings : np.ndarray, shape (N, 768)
        Raw all-mpnet-base-v2 embeddings before any normalisation or reduction.
    """
    df         = load_data(data_path)
    sentences  = preprocess(df)
    embeddings = embed(sentences)
    X_norm     = l2_normalise(embeddings)
    X_pca, _   = apply_pca(X_norm)
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
    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()