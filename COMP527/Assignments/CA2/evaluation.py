# evaluation.py
"""
Evaluate clustering results.

Usage:
    python3 evaluation.py data_train.txt label.txt

Optional:
    python3 evaluation.py data_train.txt label.txt true_labels.txt

This script:
1. Loads texts and predicted labels
2. Rebuilds embeddings from cleaned text
3. Computes internal clustering metrics:
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
4. Optionally computes external metrics if true labels are provided:
   - Adjusted Rand Index
   - Homogeneity Score
5. Displays top terms and sample texts per cluster
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
)
from sentence_transformers import SentenceTransformer

from preprocessing import clean_text, load_texts


def load_labels(file_path: str) -> np.ndarray:
    """Load one label per line."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    return np.array(labels)


def get_embeddings(texts: List[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """Generate sentence embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def print_internal_metrics(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """Print internal clustering metrics."""
    sil = silhouette_score(embeddings, labels)
    dbi = davies_bouldin_score(embeddings, labels)
    chi = calinski_harabasz_score(embeddings, labels)

    print("\nClustering Evaluation Metrics")
    print("-" * 40)
    print(f"Silhouette Score      : {sil:.6f}")
    print(f"Davies-Bouldin Index  : {dbi:.6f}")
    print(f"Calinski-Harabasz     : {chi:.6f}")


def print_external_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> None:
    """Print external clustering metrics."""
    ari = adjusted_rand_score(true_labels, pred_labels)
    hs = homogeneity_score(true_labels, pred_labels)

    print("\nExternal Evaluation Metrics")
    print("-" * 40)
    print(f"Adjusted Rand Index   : {ari:.6f}")
    print(f"Homogeneity Score     : {hs:.6f}")


def print_cluster_sizes(labels: np.ndarray) -> None:
    """Print number of documents in each cluster."""
    counts = Counter(labels)
    print("\nCluster Sizes")
    print("-" * 40)
    for cluster_id in sorted(counts):
        print(f"Cluster {cluster_id}: {counts[cluster_id]} documents")


def print_top_terms_per_cluster(cleaned_texts: List[str], labels: np.ndarray, top_n: int = 10) -> None:
    """Print top terms per cluster using CountVectorizer."""
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(cleaned_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    print("\nTop Terms Per Cluster")
    print("-" * 40)

    for cluster_id in sorted(set(labels)):
        cluster_mask = labels == cluster_id
        cluster_term_sums = np.asarray(X[cluster_mask].sum(axis=0)).ravel()
        top_indices = cluster_term_sums.argsort()[::-1][:top_n]
        top_terms = feature_names[top_indices]

        print(f"\nCluster {cluster_id}:")
        print(", ".join(top_terms))


def print_sample_documents(raw_texts: List[str], labels: np.ndarray, samples_per_cluster: int = 3) -> None:
    """Print sample documents for each cluster."""
    print("\nSample Documents Per Cluster")
    print("-" * 40)

    for cluster_id in sorted(set(labels)):
        print(f"\nCluster {cluster_id}:")
        cluster_docs = [text for text, label in zip(raw_texts, labels) if label == cluster_id]

        for i, doc in enumerate(cluster_docs[:samples_per_cluster], start=1):
            print(f"  {i}. {doc[:200]}")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage:")
        print("  python3 evaluation.py data_train.txt label.txt")
        print("  python3 evaluation.py data_train.txt label.txt true_labels.txt")
        sys.exit(1)

    text_file = sys.argv[1]
    label_file = sys.argv[2]
    true_label_file: Optional[str] = sys.argv[3] if len(sys.argv) == 4 else None

    raw_texts = load_texts(text_file)
    pred_labels = load_labels(label_file)

    if len(raw_texts) != len(pred_labels):
        print("Error: number of texts and number of predicted labels do not match.")
        sys.exit(1)

    cleaned_texts = [clean_text(text) for text in raw_texts]

    print(f"Loaded {len(raw_texts)} texts")
    print(f"Loaded {len(pred_labels)} predicted labels")

    print("Generating embeddings for evaluation...")
    embeddings = get_embeddings(cleaned_texts)

    print_internal_metrics(embeddings, pred_labels)

    if true_label_file is not None:
        true_labels = load_labels(true_label_file)
        if len(true_labels) != len(pred_labels):
            print("Error: true labels and predicted labels do not match in length.")
            sys.exit(1)
        print_external_metrics(true_labels, pred_labels)

    print_cluster_sizes(pred_labels)
    print_top_terms_per_cluster(cleaned_texts, pred_labels, top_n=10)
    print_sample_documents(raw_texts, pred_labels, samples_per_cluster=3)


if __name__ == "__main__":
    main()