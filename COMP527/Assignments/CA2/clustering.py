# clustering.py
"""
clustering.py
─────────────
Runs the full clustering pipeline on a dataset in the format:

    ID<TAB>Sentence

Usage
-----
    python clustering.py data.txt

Full pipeline
-------------
    preprocessing.py : embed  →  L2 normalise  →  PCA (100 components)
    clustering.py    : AHC (Euclidean + Ward)  →  silhouette  →  outputs

AHC configuration rationale
────────────────────────────
metric  = Euclidean
    The feature matrix produced by preprocessing is the direct PCA output —
    variance-weighted, not re-normalised.  Euclidean distance in this space
    automatically gives more weight to the earlier, higher-variance principal
    components, which carry the most semantic information.  This makes
    Euclidean the correct and natural distance metric here.

linkage = Ward
    Ward linkage merges the pair of clusters whose union minimises the
    increase in total within-cluster variance.  This produces compact,
    balanced clusters and is robust against the chaining problem that
    afflicts average linkage on text data (where chaining consistently
    collapses results to K=2).  Ward is defined only for Euclidean space,
    which is consistent with the chosen metric.

Silhouette scoring
──────────────────
Computed with the same Euclidean metric in the same PCA feature space used
for clustering, ensuring a fully consistent and honest quality measure.
Silhouette scores in text clustering with BERT-family embeddings are
typically in the range 0.05–0.20 (Petukhova et al., 2024), so values in
this range should be interpreted as realistic rather than poor.

Outputs
-------
    label.txt              – one 1-indexed cluster label per data instance
    clustering_results.csv – sentence, cluster, algorithm per row
    silhouette_scores.csv  – silhouette coefficient for K = 1 … 10
    silhouette_plot.png    – line chart of silhouette coefficient vs K
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from preprocessing import preprocess_pipeline


# ── Config ────────────────────────────────────────────────────────────────────
K_RANGE     = range(1, 11)
LABELS_OUT  = "label.txt"
RESULTS_OUT = "clustering_results.csv"
SCORES_OUT  = "silhouette_scores.csv"
PLOT_OUT    = "silhouette_plot.png"
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_ahc(
    X: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """
    Fit AHC for K = 1 … 10 and return the best result by silhouette score.

    Parameters
    ----------
    X : np.ndarray, shape (N, 100)
        Variance-weighted PCA feature matrix from preprocessing.

    Returns
    -------
    best_labels : ndarray or None   – 0-indexed cluster assignments for best K
    best_score  : float             – highest silhouette score found
    best_k      : int or None       – K that achieved best_score
    score_rows  : list of dicts     – per-K records for CSV and plot output
    """
    best_score  = -1.0
    best_labels = None
    best_k      = None
    score_rows: List[dict] = []

    print("\nEvaluating Agglomerative Hierarchical Clustering (AHC)")
    print("Configuration : metric=euclidean, linkage=ward")
    print("Silhouette    : euclidean distance, same PCA space as clustering\n")

    for k in K_RANGE:
        if k == 1:
            # Silhouette requires at least 2 clusters
            print(f"  K={k:2d}  silhouette = N/A  (undefined for a single cluster)")
            score_rows.append({"algorithm": "AHC", "k": k, "silhouette": np.nan})
            continue

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        )
        labels = model.fit_predict(X)
        score  = silhouette_score(X, labels, metric="euclidean")
        print(f"  K={k:2d}  silhouette = {score:.6f}")

        if score > best_score:
            best_score  = score
            best_labels = labels.copy()
            best_k      = k

        score_rows.append({"algorithm": "AHC", "k": k, "silhouette": score})

    return best_labels, best_score, best_k, score_rows


def save_labels(labels: np.ndarray, path: str = LABELS_OUT) -> None:
    """
    Write one cluster label per line to label.txt.
    Labels are 1-indexed (1 … K) to match the assignment specification.
    """
    np.savetxt(path, labels + 1, fmt="%d")
    print(f"\n[save] Labels             -> {path}")


def save_results(
    sentences: List[str],
    labels: np.ndarray,
    path: str = RESULTS_OUT,
) -> None:
    """Write sentence, 1-indexed cluster, and algorithm name to CSV."""
    df = pd.DataFrame({
        "sentence":  sentences,
        "cluster":   labels + 1,
        "algorithm": "AHC",
    })
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Clustering results  -> {path}")


def save_scores(score_rows: List[dict], path: str = SCORES_OUT) -> None:
    """Write per-K silhouette scores to CSV."""
    pd.DataFrame(score_rows).to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Silhouette scores   -> {path}")


def plot_silhouette_scores(score_rows: List[dict], path: str = PLOT_OUT) -> None:
    """
    Plot silhouette coefficient vs K and save as a PNG file.
    K=1 is shown on the x-axis but annotated 'N/A' because silhouette is
    undefined for a single cluster.
    """
    df    = pd.DataFrame(score_rows).sort_values("k")
    valid = df.dropna(subset=["silhouette"])

    plt.figure(figsize=(10, 6))
    plt.plot(valid["k"], valid["silhouette"], marker="o", linewidth=2, label="AHC")
    plt.annotate(
        "N/A",
        xy=(1, 0),
        xycoords=("data", "axes fraction"),
        ha="center", va="bottom",
        fontsize=9, color="grey",
    )
    plt.xlabel("K (Number of Clusters)")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis — AHC (Euclidean + Ward), K = 1 to 10")
    plt.xticks(list(K_RANGE))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] Silhouette plot      -> {path}")


def print_cluster_samples(sentences: List[str], labels: np.ndarray, n: int = 5) -> None:
    """Print the first n sentences from each cluster (1-indexed labels)."""
    df = pd.DataFrame({"sentence": sentences, "cluster": labels + 1})
    print("\nSample documents per cluster")
    print("-" * 40)
    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        print(f"\nCluster {cid}  ({len(subset)} documents)")
        for i, text in enumerate(subset["sentence"].head(n).tolist(), start=1):
            print(f"  {i}. {text[:160]}")


def main() -> None:
    # Accept either no argument (defaults to data_train.txt) or exactly one
    # filename argument.  More than one argument is a usage error.
    if len(sys.argv) == 1:
        data_path = "data_train.txt"
        print(f"No data file specified — using default: {data_path}")
    elif len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        print("Usage: python clustering.py [data.txt]")
        sys.exit(1)

    if not Path(data_path).exists():
        print(f"Error: file not found -> {data_path}")
        sys.exit(1)

    # ── Preprocessing: embed → L2 normalise → PCA ─────────────────────────
    X, sentences, embeddings = preprocess_pipeline(data_path)
    print(f"\n[features] Shape      : {X.shape}")
    print(f"[features] Sentences  : {len(sentences)}")
    print(f"[features] Embeddings : {embeddings.shape}")

    # ── Clustering: AHC with Euclidean + Ward ─────────────────────────────
    best_labels, best_score, best_k, score_rows = evaluate_ahc(X)

    if best_labels is None or best_k is None:
        raise RuntimeError("AHC produced no valid clustering result.")

    print("\n" + "=" * 60)
    print("Best AHC Result")
    print("=" * 60)
    print(f"  Algorithm  : AHC  (euclidean, ward)")
    print(f"  Best K     : {best_k}")
    print(f"  Silhouette : {best_score:.6f}")

    # ── Save all outputs ──────────────────────────────────────────────────
    print()
    save_labels(best_labels)
    save_results(sentences, best_labels)
    save_scores(score_rows)
    plot_silhouette_scores(score_rows)
    print_cluster_samples(sentences, best_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()