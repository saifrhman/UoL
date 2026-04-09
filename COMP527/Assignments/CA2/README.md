# CA2 Text Clustering Pipeline

## Overview

This project implements an end-to-end text clustering pipeline for grouping semantically similar sentences without predefined labels. The task focuses on unsupervised learning, where the goal is to discover natural structure in a sentence dataset rather than predict known classes.

Sentences are encoded using MiniLM, transformed through L2 normalisation and PCA, and then clustered using Agglomerative Hierarchical Clustering (AHC). The best number of clusters is selected automatically based on silhouette score.

## Features

- Loads sentence data from a tab-separated text file
- Cleans and standardises raw sentence text while preserving row alignment
- Generates sentence embeddings using `sentence-transformers/all-MiniLM-L12-v2`
- Applies L2 normalisation and PCA (50 components) for feature transformation
- Clusters with AHC (Euclidean distance, Ward linkage) for K = 2 to 10
- Selects the best K automatically using silhouette score
- Exports cluster labels, detailed results, and evaluation plots

## Dataset Format

The input dataset must be a UTF-8 tab-separated file with the following columns:

```text
ID<TAB>Sentence
```

Example:

```text
ID	Sentence
1	I thought I would give them another try since I'd never ordered a Sicilian pizza from there.
2	CORIOLANUS: Why then should I be consul?
```

## Methodology

### 1. Preprocessing (`preprocessing.py`)

The preprocessing pipeline runs in the following stages:

- Loads the dataset from a tab-separated file
- Cleans sentence text by removing escaped newlines, escaped quotes, and redundant whitespace
- Preserves sentence order to ensure output labels align with the original data
- Encodes each sentence into a 384-dimensional dense embedding using MiniLM
- Applies L2 normalisation so that PCA decomposes directional (semantic) variance rather than magnitude variance
- Applies PCA to reduce dimensionality from 384 to 50 components, retaining approximately 56% of the variance

The PCA output is intentionally left un-normalised. PCA assigns more variance to earlier components, and Euclidean distance in this space naturally weights the most informative components more heavily — erasing this structure by re-normalising would degrade cluster separation.

### 2. Clustering (`clustering.py`)

Clustering is performed using **Agglomerative Hierarchical Clustering (AHC)** with:

- `metric = euclidean` — consistent with the variance-weighted PCA feature space, giving more weight to higher-variance (more informative) components automatically
- `linkage = ward` — merges clusters to minimise the increase in total within-cluster variance, producing compact and balanced clusters; defined only for Euclidean space

AHC is evaluated for K = 2 to 10. K = 1 is skipped because silhouette score is undefined for a single cluster. The K with the highest silhouette score is selected as the final result.

### 3. Evaluation

Clustering quality is measured using the **silhouette score**, computed with Euclidean distance in the same PCA feature space used for clustering. This ensures a fully consistent and honest quality measure. Silhouette scores in text clustering with BERT-family embeddings are typically in the range 0.05–0.20, so values in this range should be interpreted as realistic rather than poor.

## Installation

Recommended environment:

- Python 3.10+

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the full clustering pipeline with:

```bash
python clustering.py data_train.txt
```

## Outputs

| File | Description |
|---|---|
| `label.txt` | Final cluster assignment (1-indexed) for each input sentence, one label per line |
| `clustering_results.csv` | Each sentence with its assigned cluster and algorithm name |
| `silhouette_scores.csv` | Silhouette coefficient recorded for each K = 1 to 10 |
| `silhouette_plot.png` | Line chart of silhouette coefficient vs K |

## Project Structure

```text
CA2/
├── clustering.py
├── preprocessing.py
├── data_train.txt
├── requirements.txt
└── README.md
```

## Requirements

Libraries used in this project:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `sentence-transformers`
- `torch`

## Notes

- This project uses only local Python libraries and pretrained models
- No cloud APIs or external hosted ML services are used
- The pipeline is suitable for offline academic experimentation, subject to local model and package installation

## Author

Saif Ur Rehman
