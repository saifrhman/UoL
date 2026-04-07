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
2. Runs clustering algorithms on the transformed features
3. Evaluates silhouette scores for K = 1 to 10
4. Selects the best clustering result
5. Saves:
   - label.txt
   - clustering_results.csv
   - silhouette_scores.csv
   - silhouette_plot.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from preprocessing import preprocess_pipeline

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[warn] hdbscan not installed. HDBSCAN will be skipped.")

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("[warn] scikit-learn-extra not installed. KMedoids will be skipped.")


# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
K_RANGE = range(1, 11)

LABELS_OUT = "label.txt"
RESULTS_OUT = "clustering_results.csv"
SCORES_OUT = "silhouette_scores.csv"
PLOT_OUT = "silhouette_plot.png"
# ─────────────────────────────────────────────────────────────────────────────


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean"
) -> Optional[float]:
    """
    Compute silhouette score safely.
    Returns None if silhouette is not defined.
    """
    unique_labels = set(labels)

    if -1 in unique_labels:
        unique_labels.discard(-1)

    if len(unique_labels) < 2:
        return None

    if -1 in labels:
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return None
        return silhouette_score(X[mask], labels[mask], metric=metric)

    return silhouette_score(X, labels, metric=metric)


def evaluate_kmeans_random(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """Evaluate KMeans with random initialization for K = 1..10."""
    best_score = -1.0
    best_labels = None
    best_k = None
    score_rows: List[dict] = []

    print("\nEvaluating KMeans (random init)")
    for k in K_RANGE:
        if k == 1:
            print(f"K={k}, silhouette=N/A")
            score_rows.append({
                "algorithm": "kmeans",
                "k": k,
                "silhouette": np.nan
            })
            continue

        model = KMeans(
            n_clusters=k,
            init="random",
            n_init=10,
            random_state=RANDOM_STATE,
            max_iter=500,
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="euclidean")
        print(f"K={k}, silhouette={score:.6f}")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

        score_rows.append({
            "algorithm": "kmeans",
            "k": k,
            "silhouette": score
        })

    return best_labels, best_score, best_k, score_rows


def evaluate_kmeans_plus(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """Evaluate KMeans++ for K = 1..10."""
    best_score = -1.0
    best_labels = None
    best_k = None
    score_rows: List[dict] = []

    print("\nEvaluating KMeans++")
    for k in K_RANGE:
        if k == 1:
            print(f"K={k}, silhouette=N/A")
            score_rows.append({
                "algorithm": "kmeans++",
                "k": k,
                "silhouette": np.nan
            })
            continue

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
            max_iter=500,
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="euclidean")
        print(f"K={k}, silhouette={score:.6f}")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

        score_rows.append({
            "algorithm": "kmeans++",
            "k": k,
            "silhouette": score
        })

    return best_labels, best_score, best_k, score_rows


def evaluate_ahc(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """Evaluate Agglomerative Hierarchical Clustering for K = 1..10."""
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


def evaluate_hdbscan(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str], List[dict]]:
    """
    Evaluate HDBSCAN once.
    HDBSCAN does not use a fixed K, so it is evaluated separately.
    """
    score_rows: List[dict] = []

    if not HDBSCAN_AVAILABLE:
        print("\nHDBSCAN not installed. Skipping.")
        return None, None, None, score_rows

    print("\nEvaluating HDBSCAN")
    model = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = float(np.mean(labels == -1) * 100)
    score = compute_silhouette(X, labels, metric="euclidean")

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {noise_pct:.2f}%")
    print(f"Silhouette: {score if score is not None else 'N/A'}")

    score_rows.append({
        "algorithm": "HDBSCAN",
        "k": n_clusters,
        "silhouette": score if score is not None else np.nan
    })

    detail = f"clusters={n_clusters}, noise={noise_pct:.2f}%"
    return labels, score, detail, score_rows


def evaluate_kmedoids(
    X: np.ndarray
) -> Tuple[Optional[np.ndarray], float, Optional[int], List[dict]]:
    """Evaluate KMedoids for K = 1..10."""
    score_rows: List[dict] = []

    if not KMEDOIDS_AVAILABLE:
        print("\nKMedoids not installed. Skipping.")
        return None, -1.0, None, score_rows

    best_score = -1.0
    best_labels = None
    best_k = None

    print("\nEvaluating KMedoids")
    for k in K_RANGE:
        if k == 1:
            print(f"K={k}, silhouette=N/A")
            score_rows.append({
                "algorithm": "kmedoids",
                "k": k,
                "silhouette": np.nan
            })
            continue

        model = KMedoids(
            n_clusters=k,
            metric="cosine",
            init="k-medoids++",
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        print(f"K={k}, silhouette={score:.6f}")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

        score_rows.append({
            "algorithm": "kmedoids",
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
    algorithm: str,
    path: str = RESULTS_OUT
) -> None:
    """Save detailed clustering results."""
    df = pd.DataFrame({
        "sentence": sentences,
        "cluster": labels,
        "algorithm": algorithm,
    })
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Detailed clustering results saved to {path}")


def save_scores(score_rows: List[dict], path: str = SCORES_OUT) -> None:
    """Save silhouette scores for all algorithms."""
    df = pd.DataFrame(score_rows)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Silhouette scores saved to {path}")


def plot_silhouette_scores(score_rows: List[dict], path: str = PLOT_OUT) -> None:
    """
    Save silhouette plot for K = 1..10.
    HDBSCAN is excluded because it does not use fixed K.
    """
    df = pd.DataFrame(score_rows)
    df_fixed = df[df["algorithm"] != "HDBSCAN"].copy()

    plt.figure(figsize=(10, 6))

    for algo in df_fixed["algorithm"].unique():
        subset = df_fixed[df_fixed["algorithm"] == algo].sort_values("k")
        plt.plot(
            subset["k"],
            subset["silhouette"],
            marker="o",
            label=algo
        )

    plt.xlabel("K")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis for K = 1 to 10")
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
        cluster_name = "NOISE" if cid == -1 else f"Cluster {cid}"
        print(f"\n{cluster_name}: {len(subset)} documents")

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

    all_score_rows: List[dict] = []
    results: Dict[str, Dict] = {}

    # KMeans random
    km_labels, km_score, km_k, km_rows = evaluate_kmeans_random(X)
    all_score_rows.extend(km_rows)
    if km_labels is not None:
        results["kmeans"] = {
            "labels": km_labels,
            "score": km_score,
            "detail": f"k={km_k}",
        }

    # KMeans++
    kmpp_labels, kmpp_score, kmpp_k, kmpp_rows = evaluate_kmeans_plus(X)
    all_score_rows.extend(kmpp_rows)
    if kmpp_labels is not None:
        results["kmeans++"] = {
            "labels": kmpp_labels,
            "score": kmpp_score,
            "detail": f"k={kmpp_k}",
        }

    # AHC
    ahc_labels, ahc_score, ahc_k, ahc_rows = evaluate_ahc(X)
    all_score_rows.extend(ahc_rows)
    if ahc_labels is not None:
        results["AHC"] = {
            "labels": ahc_labels,
            "score": ahc_score,
            "detail": f"k={ahc_k}",
        }

    # HDBSCAN
    hdb_labels, hdb_score, hdb_detail, hdb_rows = evaluate_hdbscan(X)
    all_score_rows.extend(hdb_rows)
    if hdb_labels is not None and hdb_score is not None:
        results["HDBSCAN"] = {
            "labels": hdb_labels,
            "score": hdb_score,
            "detail": hdb_detail,
        }

    # KMedoids
    kmedoids_labels, kmedoids_score, kmedoids_k, kmedoids_rows = evaluate_kmedoids(X)
    all_score_rows.extend(kmedoids_rows)
    if kmedoids_labels is not None:
        results["kmedoids"] = {
            "labels": kmedoids_labels,
            "score": kmedoids_score,
            "detail": f"k={kmedoids_k}",
        }

    if not results:
        raise RuntimeError("No clustering algorithm produced a valid result.")

    best_algorithm = max(results, key=lambda name: results[name]["score"])
    best_labels = results[best_algorithm]["labels"]
    best_score = results[best_algorithm]["score"]
    best_detail = results[best_algorithm]["detail"]

    print("\n" + "=" * 60)
    print("Best algorithm")
    print("=" * 60)
    print(f"Algorithm : {best_algorithm}")
    print(f"Score     : {best_score:.6f}")
    print(f"Details   : {best_detail}")

    save_labels(best_labels)
    save_results(sentences, best_labels, best_algorithm)
    save_scores(all_score_rows)
    plot_silhouette_scores(all_score_rows)
    print_cluster_samples(sentences, best_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()