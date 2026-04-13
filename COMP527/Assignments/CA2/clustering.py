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
    preprocessing.py : embed (all-mpnet-base-v2, 768-dim)
                       → L2 normalise
                       → Adaptive PCA (data-driven variance threshold)
                       → UMAP (non-linear manifold projection, 20-dim)

    clustering.py    : AHC (Euclidean + Ward)
                       → silhouette analysis (K = 1 … 10)
                       → outputs

AHC configuration rationale
────────────────────────────
metric  = Euclidean
    UMAP output lies in an unconstrained Euclidean space (points are NOT
    unit-normalised after UMAP).  Ward linkage is mathematically defined
    only for Euclidean distance, and the UMAP projection preserves cluster
    topology in this space, making Euclidean distances semantically
    meaningful for AHC.

linkage = Ward
    Ward linkage merges the pair of clusters whose union minimises the
    increase in total within-cluster sum of squares (WCSS).  This produces
    compact, balanced clusters and is robust against the chaining artefact
    that afflicts single and average linkage on text data.  Ward is the
    standard linkage for AHC in text clustering benchmarks (Petukhova et
    al., 2024).

Silhouette scoring
──────────────────
Computed with the same Euclidean metric in the same UMAP feature space used
for clustering — a fully consistent internal validity measure.  Silhouette
scores for text clustering with BERT-family embeddings plus UMAP are
typically higher (0.15–0.50) than those without UMAP (0.05–0.20) because
UMAP sharpens cluster boundaries in the projected space (Petukhova et al.,
2024; Vizuara, 2024).

Calinski-Harabasz Index
────────────────────────
The Calinski-Harabasz Index (CHI) is reported alongside the silhouette score
as a complementary internal validity metric.  CHI = Tr(Bk) / Tr(Wk) × (N-k)
/ (k-1), where Tr(Bk) is the between-cluster dispersion and Tr(Wk) the
within-cluster dispersion.  Higher is better.  CHI tends to favour compact,
well-separated clusters and provides a second line of evidence for the
optimal K.

Outputs
-------
    label.txt              – one 1-indexed cluster label per data instance
    clustering_results.csv – sentence, cluster, algorithm per row
    silhouette_scores.csv  – silhouette coefficient and CHI for K = 1 … 10
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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
    X : np.ndarray, shape (N, n_features)
        Feature matrix from preprocessing (UMAP output).  Ward linkage with
        Euclidean distance is appropriate because UMAP output lies in an
        unconstrained Euclidean space.

    Returns
    -------
    best_labels : ndarray or None   – 0-indexed cluster assignments for best K
    best_score  : float             – highest silhouette score found
    best_k      : int or None       – K that achieved best_score
    score_rows  : list of dicts     – per-K records for CSV and plot output
                                      (includes both silhouette and CHI)
    """
    best_score  = -1.0
    best_labels = None
    best_k      = None
    score_rows: List[dict] = []

    print("\nEvaluating Agglomerative Hierarchical Clustering (AHC)")
    print("Configuration : metric=euclidean, linkage=ward")
    print("Evaluation    : silhouette score + Calinski-Harabasz Index\n")

    for k in K_RANGE:
        if k == 1:
            # Both silhouette and CHI require at least 2 clusters
            print(f"  K={k:2d}  silhouette = N/A  CHI = N/A  "
                  f"(undefined for a single cluster)")
            score_rows.append({
                "algorithm": "AHC", "k": k,
                "silhouette": np.nan, "calinski_harabasz": np.nan
            })
            continue

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        )
        labels = model.fit_predict(X)
        sil    = silhouette_score(X, labels, metric="euclidean")
        chi    = calinski_harabasz_score(X, labels)
        print(f"  K={k:2d}  silhouette = {sil:.6f}   CHI = {chi:10.2f}")

        if sil > best_score:
            best_score  = sil
            best_labels = labels.copy()
            best_k      = k

        score_rows.append({
            "algorithm": "AHC", "k": k,
            "silhouette": sil, "calinski_harabasz": chi
        })

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
    """Write per-K silhouette and CHI scores to CSV."""
    pd.DataFrame(score_rows).to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Silhouette scores   -> {path}")


def plot_silhouette_scores(score_rows: List[dict], path: str = PLOT_OUT) -> None:
    """
    Plot silhouette coefficient (primary axis) and Calinski-Harabasz Index
    (secondary axis) vs K, then save as a PNG file.
    K=1 is shown on the x-axis but annotated 'N/A' because both metrics are
    undefined for a single cluster.
    """
    df    = pd.DataFrame(score_rows).sort_values("k")
    valid = df.dropna(subset=["silhouette"])

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Primary axis: silhouette
    color_sil = "#1f77b4"
    ax1.plot(valid["k"], valid["silhouette"],
             marker="o", linewidth=2.5, color=color_sil, label="Silhouette (left)")
    ax1.set_xlabel("K (Number of Clusters)", fontsize=12)
    ax1.set_ylabel("Silhouette Coefficient", color=color_sil, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_sil)
    ax1.annotate(
        "N/A", xy=(1, 0), xycoords=("data", "axes fraction"),
        ha="center", va="bottom", fontsize=9, color="grey",
    )

    # Secondary axis: Calinski-Harabasz
    ax2 = ax1.twinx()
    color_chi = "#ff7f0e"
    ax2.plot(valid["k"], valid["calinski_harabasz"],
             marker="s", linewidth=2, linestyle="--",
             color=color_chi, label="Calinski-Harabasz (right)")
    ax2.set_ylabel("Calinski-Harabasz Index", color=color_chi, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_chi)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.title(
        "Silhouette & Calinski-Harabasz Analysis\n"
        "AHC (Euclidean + Ward) · all-mpnet-base-v2 + Adaptive PCA + UMAP\n"
        "K = 1 to 10",
        fontsize=12,
    )
    ax1.set_xticks(list(K_RANGE))
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] Silhouette plot      -> {path}")


def print_cluster_samples(sentences: List[str], labels: np.ndarray, n: int = 5) -> None:
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

    # ── Preprocessing: embed → L2 normalise → adaptive PCA → UMAP ────────
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