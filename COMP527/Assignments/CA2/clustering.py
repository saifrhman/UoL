# clustering.py

import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import load_data, preprocess

file_path = '/home/saif/Projects/UoL/COMP527/Assignments/CA2/data_train.txt'
def main(file_path):
    sentences = load_data(file_path)
    X = preprocess(sentences)

    silhouette_scores = []

    K_range = range(2, 11)  # silhouette not valid for K=1

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score}")

    # Plot
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.savefig('silhouette.png')
    plt.show()

    # Choose best K
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print("Best K:", best_k)

    # Final clustering
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_kmeans.fit_predict(X)

    # Save labels
    with open('label.txt', 'w') as f:
        for label in final_labels:
            f.write(str(label) + '\n')


if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)