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
    preprocessing.py : embed (all-mpnet-base-v2, 768-dim, all rows)
                       -> L2 normalise
                       -> Adaptive PCA (80 % variance threshold)
                       -> UMAP (cosine, n_components=5, min_dist=0.0,
                                n_neighbors=10)

    clustering.py    : silhouette sweep K=1..10
                       -> best K selected automatically by silhouette score
                       -> AHC (Ward, Euclidean) at best K
                       -> outputs

Chosen K rationale
------------------
The optimal K is selected automatically as the value in K = 2 ... 10 that
produces the highest silhouette coefficient.  No K is hardcoded.  The
silhouette sweep and the final clustering are fully consistent: the same AHC
configuration (Ward linkage, Euclidean distance, 5-dim UMAP feature space)
is used for both.

AHC configuration
-----------------
metric  = Euclidean  (correct for UMAP output space; Ward requires Euclidean)
linkage = Ward       (minimises within-cluster variance; robust to chaining)

Silhouette
----------
Computed in the same 5-dim UMAP Euclidean space as clustering -- fully
self-consistent.

Outputs
-------
    label.txt              -- one 1-indexed cluster label per data instance
    clustering_results.csv -- sentence, cluster, algorithm per row
    silhouette_scores.csv  -- silhouette coefficient for K = 1 ... 10
    silhouette_plot.png    -- line chart of silhouette coefficient vs K
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
    Fit AHC for K = 1 ... 10 and return the best result by silhouette score.

    The best K is selected automatically as the K in {2, ..., 10} that
    maximises the silhouette coefficient.  No K is hardcoded.

    Returns
    -------
    best_labels : 0-indexed cluster assignments for best K
    best_score  : highest silhouette score found
    best_k      : K that achieved best_score
    score_rows  : per-K records for CSV and plot
    """
    best_score  = -1.0
    best_labels = None
    best_k      = None
    score_rows: List[dict] = []

    print("\nEvaluating Agglomerative Hierarchical Clustering (AHC)")
    print("Configuration : metric=euclidean, linkage=ward")
    print("Feature space : 5-dim UMAP (cosine neighbourhood graph)\n")

    for k in K_RANGE:
        if k == 1:
            print(f"  K={k:2d}  silhouette = N/A  (undefined for a single cluster)")
            score_rows.append({"algorithm": "AHC", "k": k, "silhouette": np.nan})
            continue

        labels = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        ).fit_predict(X)
        score = silhouette_score(X, labels, metric="euclidean")
        print(f"  K={k:2d}  silhouette = {score:.6f}")

        if score > best_score:
            best_score  = score
            best_labels = labels.copy()
            best_k      = k

        score_rows.append({"algorithm": "AHC", "k": k, "silhouette": score})

    return best_labels, best_score, best_k, score_rows


def save_labels(labels: np.ndarray, path: str = LABELS_OUT) -> None:
    """Write one 1-indexed cluster label per line to label.txt."""
    np.savetxt(path, labels + 1, fmt="%d")
    print(f"\n[save] Labels             -> {path}  ({len(labels):,} rows)")


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


def plot_silhouette_scores(
    score_rows: List[dict],
    best_k: int,
    path: str = PLOT_OUT,
) -> None:
    """
    Plot silhouette coefficient vs K.
    K=1 is annotated 'N/A'; best_k is marked with a vertical dashed line.
    The position of the line is determined automatically from the sweep
    results -- nothing is hardcoded.
    """
    df    = pd.DataFrame(score_rows).sort_values("k")
    valid = df.dropna(subset=["silhouette"])

    plt.figure(figsize=(10, 6))
    plt.plot(valid["k"], valid["silhouette"], marker="o", linewidth=2, label="AHC")
    plt.axvline(x=best_k, color="tomato", linestyle="--", linewidth=1.5,
                label=f"Best K = {best_k} (data-driven)")
    plt.annotate(
        "N/A",
        xy=(1, 0),
        xycoords=("data", "axes fraction"),
        ha="center", va="bottom",
        fontsize=9, color="grey",
    )
    plt.xlabel("K (Number of Clusters)")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis -- AHC (Euclidean + Ward), K = 1 to 10")
    plt.xticks(list(K_RANGE))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] Silhouette plot      -> {path}")


def print_cluster_samples(
    sentences: List[str],
    labels: np.ndarray,
    n: int = 5,
) -> None:
    """Print the first n sentences from each cluster (1-indexed labels)."""
    df = pd.DataFrame({"sentence": sentences, "cluster": labels + 1})
    print("\nSample documents per cluster")
    print("-" * 50)
    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        pct    = 100.0 * len(subset) / len(df)
        print(f"\nCluster {cid}  ({len(subset):,} documents, {pct:.1f}%)")
        for i, text in enumerate(subset["sentence"].head(n).tolist(), start=1):
            print(f"  {i}. {text[:160]}")


def main() -> None:
    if len(sys.argv) == 1:
        data_path = "data_train.txt"
        print(f"No data file specified -- using default: {data_path}")
    elif len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        print("Usage: python clustering.py [data.txt]")
        sys.exit(1)

    if not Path(data_path).exists():
        print(f"Error: file not found -> {data_path}")
        sys.exit(1)

    # ── Preprocessing ──────────────────────────────────────────────────────
    X, sentences, embeddings = preprocess_pipeline(data_path)
    print(f"\n[features] Shape      : {X.shape}")
    print(f"[features] Sentences  : {len(sentences)}")
    print(f"[features] Embeddings : {embeddings.shape}")

    # ── Silhouette sweep + best K selection ───────────────────────────────
    best_labels, best_score, best_k, score_rows = evaluate_ahc(X)

    if best_labels is None or best_k is None:
        raise RuntimeError("AHC produced no valid clustering result.")

    print("\n" + "=" * 60)
    print("Best AHC Result  (data-driven)")
    print("=" * 60)
    print(f"  Algorithm  : AHC  (euclidean, ward)")
    print(f"  Best K     : {best_k}")
    print(f"  Silhouette : {best_score:.6f}")

    # ── Save all outputs ───────────────────────────────────────────────────
    print()
    save_labels(best_labels)
    save_results(sentences, best_labels)
    save_scores(score_rows)
    plot_silhouette_scores(score_rows, best_k)
    print_cluster_samples(sentences, best_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()