# clustering.py
"""
clustering.py
─────────────
Runs the full clustering pipeline on a dataset in the format:

    ID<TAB>Sentence

Usage
-----
python clustering.py data.txt

What this script does
---------------------
1. Calls the preprocessing pipeline from preprocessing.py
2. Runs Agglomerative Hierarchical Clustering (AHC) on the transformed features
3. Evaluates silhouette scores for K = 1 to 10
4. Selects the best AHC result
5. Saves:
   - label.txt
   - clustering_results.csv
   - silhouette_scores.csv
   - silhouette_plot.png
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
K_RANGE = range(1, 11)

LABELS_OUT = "label.txt"
RESULTS_OUT = "clustering_results.csv"
SCORES_OUT = "silhouette_scores.csv"
PLOT_OUT = "silhouette_plot.png"
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_ahc(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """
    Evaluate Agglomerative Hierarchical Clustering (AHC) for K = 1..10.

    Returns
    -------
    best_labels : np.ndarray or None
        Cluster labels for the best K.
    best_score : float
        Best silhouette score found.
    best_k : int or None
        K corresponding to the best silhouette score.
    score_rows : List[dict]
        Rows containing K and silhouette values for saving/plotting.
    """
    best_score = -1.0
    best_labels = None
    best_k = None
    score_rows: List[dict] = []

    print("\nEvaluating Agglomerative Hierarchical Clustering (AHC)")
    for k in K_RANGE:
        if k == 1:
            print(f"K={k}, silhouette=N/A")
            score_rows.append({
                "algorithm": "AHC",
                "k": k,
                "silhouette": np.nan
            })
            continue

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        print(f"K={k}, silhouette={score:.6f}")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

        score_rows.append({
            "algorithm": "AHC",
            "k": k,
            "silhouette": score
        })

    return best_labels, best_score, best_k, score_rows


def save_labels(labels: np.ndarray, path: str = LABELS_OUT) -> None:
    """Save final labels as single-column label.txt."""
    np.savetxt(path, labels, fmt="%d")
    print(f"\n[save] Labels saved to {path}")


def save_results(
    sentences: List[str],
    labels: np.ndarray,
    path: str = RESULTS_OUT
) -> None:
    """Save detailed clustering results."""
    df = pd.DataFrame({
        "sentence": sentences,
        "cluster": labels,
        "algorithm": "AHC",
    })
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Detailed clustering results saved to {path}")


def save_scores(score_rows: List[dict], path: str = SCORES_OUT) -> None:
    """Save silhouette scores for AHC."""
    df = pd.DataFrame(score_rows)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Silhouette scores saved to {path}")


def plot_silhouette_scores(score_rows: List[dict], path: str = PLOT_OUT) -> None:
    """Save silhouette plot for AHC over K = 1..10."""
    df = pd.DataFrame(score_rows).sort_values("k")

    plt.figure(figsize=(10, 6))
    plt.plot(
        df["k"],
        df["silhouette"],
        marker="o",
        label="AHC"
    )
    plt.xlabel("K")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis for AHC (K = 1 to 10)")
    plt.xticks(list(K_RANGE))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[save] Silhouette plot saved to {path}")


def print_cluster_samples(sentences: List[str], labels: np.ndarray, n: int = 5) -> None:
    """Print sample sentences from each cluster."""
    df = pd.DataFrame({"sentence": sentences, "cluster": labels})

    print("\nSample documents per cluster")
    print("-" * 40)

    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        print(f"\nCluster {cid}: {len(subset)} documents")

        samples = subset["sentence"].head(n).tolist()
        for i, text in enumerate(samples, start=1):
            print(f"  {i}. {text[:160]}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python clustering.py data.txt")
        sys.exit(1)

    data_path = sys.argv[1]

    if not Path(data_path).exists():
        print(f"Error: file not found -> {data_path}")
        sys.exit(1)

    # Full preprocessing pipeline from preprocessing.py
    X, sentences, embeddings = preprocess_pipeline(data_path)

    print(f"\n[features] Final feature matrix shape: {X.shape}")
    print(f"[features] Number of aligned sentences: {len(sentences)}")
    print(f"[features] Raw embeddings shape: {embeddings.shape}")

    # Run AHC only
    best_labels, best_score, best_k, score_rows = evaluate_ahc(X)

    if best_labels is None or best_k is None:
        raise RuntimeError("AHC did not produce a valid clustering result.")

    print("\n" + "=" * 60)
    print("Best AHC result")
    print("=" * 60)
    print("Algorithm : AHC")
    print(f"Score     : {best_score:.6f}")
    print(f"Details   : k={best_k}")

    save_labels(best_labels)
    save_results(sentences, best_labels)
    save_scores(score_rows)
    plot_silhouette_scores(score_rows)
    print_cluster_samples(sentences, best_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()