# clustering.py
"""
Text clustering using Sentence-BERT embeddings + KMeans.

Usage:
    python3 clustering.py data_train.txt

This script:
1. Loads raw texts
2. Preprocesses them
3. Generates sentence embeddings
4. Tests K from 2 to 10 using silhouette score
5. Saves:
   - label.txt
   - silhouette_plot.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from preprocessing import clean_text, load_texts


def get_embeddings(texts: List[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """Generate sentence embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def evaluate_k_range(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Tuple[List[int], List[float], int]:
    """
    Evaluate KMeans for a range of K values using silhouette score.
    Returns:
        ks: list of K values
        scores: corresponding silhouette scores
        best_k: K with highest silhouette score
    """
    ks = []
    scores = []

    print("\nEvaluating KMeans")
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        score = silhouette_score(embeddings, labels)
        ks.append(k)
        scores.append(score)

        print(f"K={k}, silhouette={score:.6f}")

    best_index = int(np.argmax(scores))
    best_k = ks[best_index]

    return ks, scores, best_k


def save_silhouette_plot(ks: List[int], scores: List[float], output_file: str = "silhouette_plot.png") -> None:
    """Save silhouette coefficient plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis for KMeans")
    plt.xticks(ks)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved silhouette plot to {output_file}")


def save_labels(labels: np.ndarray, output_file: str = "label.txt") -> None:
    """Save cluster labels one per line."""
    with open(output_file, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Saved cluster labels to {output_file}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 clustering.py data_train.txt")
        sys.exit(1)

    input_file = sys.argv[1]

    if not Path(input_file).exists():
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    raw_texts = load_texts(input_file)
    print(f"Loaded {len(raw_texts)} texts")

    print("Preprocessing texts...")
    cleaned_texts = [clean_text(text) for text in raw_texts]

    print("Generating Sentence-BERT embeddings...")
    embeddings = get_embeddings(cleaned_texts)

    ks, scores, best_k = evaluate_k_range(embeddings, k_min=2, k_max=10)
    print(f"\nBest K selected: {best_k}")

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_model.fit_predict(embeddings)

    save_labels(final_labels, "label.txt")
    save_silhouette_plot(ks, scores, "silhouette_plot.png")


if __name__ == "__main__":
    main()