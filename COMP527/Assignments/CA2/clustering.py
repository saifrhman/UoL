from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import load_data, preprocess


SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_PATH = SCRIPT_DIR / "label.txt"
PLOT_PATH = SCRIPT_DIR / "silhouette_analysis.png"
SUMMARY_PATH = SCRIPT_DIR / "clustering_results_explanation.txt"
RANDOM_STATE = 42


def evaluate_k_values(feature_matrix, k_values: range) -> dict[int, float]:
    """Compute silhouette scores for candidate K values."""
    scores: dict[int, float] = {1: math.nan}

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels, metric="cosine")
        scores[k] = score
        print(f"K={k}, silhouette={score:.6f}")

    return scores


def save_silhouette_plot(scores: dict[int, float]) -> None:
    """Create the required silhouette analysis plot for K = 1..10."""
    x_values = list(range(1, 11))
    y_values = [scores.get(k, math.nan) for k in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2, color="#1f77b4")
    plt.xticks(x_values)
    plt.xlabel("K")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis for K = 1 to 10")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.annotate(
        "K=1 is undefined for silhouette scoring",
        xy=(1, 0),
        xytext=(1.6, min(v for v in y_values[1:] if not math.isnan(v))),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=9,
        color="#444444",
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def choose_best_k(scores: dict[int, float]) -> int:
    valid_scores = {k: v for k, v in scores.items() if k > 1 and not math.isnan(v)}
    return max(valid_scores, key=valid_scores.get)


def save_labels(labels) -> None:
    with LABEL_PATH.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")


def save_results_summary(best_k: int, scores: dict[int, float]) -> None:
    lines = [
        "Clustering algorithm: KMeans on TF-IDF text features.",
        "Silhouette scoring metric: cosine similarity.",
        f"Chosen K: {best_k}",
        "",
        "Silhouette scores:",
    ]

    for k in range(1, 11):
        value = scores.get(k, math.nan)
        display = "undefined" if math.isnan(value) else f"{value:.6f}"
        lines.append(f"K={k}: {display}")

    lines.extend(
        [
            "",
            (
                f"K={best_k} was selected because it achieved the highest silhouette "
                "score among the valid K values (2 to 10), indicating the best balance "
                "of compact clusters and separation between clusters."
            ),
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    sentences = load_data(data_file)

    if not sentences:
        raise ValueError("No data instances were loaded from the input file.")

    feature_matrix = preprocess(sentences)
    scores = evaluate_k_values(feature_matrix, range(2, 11))
    save_silhouette_plot(scores)

    best_k = choose_best_k(scores)
    final_model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    final_labels = final_model.fit_predict(feature_matrix)

    save_labels(final_labels)
    save_results_summary(best_k, scores)

    print(f"Best K selected: {best_k}")
    print(f"Label file written to: {LABEL_PATH}")
    print(f"Silhouette plot written to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
