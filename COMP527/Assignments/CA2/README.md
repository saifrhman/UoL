# CA2 Text Clustering Pipeline

## Overview

This project implements an end-to-end text clustering pipeline for grouping semantically similar sentences without predefined labels. The task focuses on unsupervised learning, where the goal is to discover natural structure in a sentence dataset rather than predict known classes.

The approach combines transformer-based sentence embeddings with dimensionality reduction and multiple clustering algorithms. Sentences are encoded using MiniLM, transformed through normalization, PCA, and UMAP, and then clustered using several algorithms to identify the best-performing configuration based on silhouette score.

## Features

- Loads sentence data from a tab-separated text file
- Cleans and standardizes raw sentence text while preserving row alignment
- Generates sentence embeddings using `sentence-transformers/all-MiniLM-L12-v2`
- Applies L2 normalization, PCA, and UMAP for feature transformation
- Evaluates multiple clustering algorithms on the same feature space
- Compares clustering quality using silhouette score
- Selects the best-performing clustering result automatically
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

### 1. Preprocessing

The preprocessing stage:

- Loads the dataset from a tab-separated file
- Cleans sentence text by removing escaped newlines, escaped quotes, and redundant whitespace
- Preserves sentence order to ensure output labels align with the original data
- Converts each sentence into a dense semantic embedding using MiniLM

### 2. Dimensionality Reduction

The embedding space is reduced in two stages:

- **PCA** is applied first to reduce dimensionality while retaining most of the variance and improving efficiency
- **UMAP** is then used to capture non-linear structure and preserve local neighbourhood relationships that are useful for clustering

This PCA-plus-UMAP combination helps produce a compact feature representation that is more suitable for downstream clustering than raw high-dimensional embeddings alone.

### 3. Clustering Algorithms

The pipeline evaluates the following clustering methods:

- KMeans with random initialization
- KMeans++
- Agglomerative Hierarchical Clustering (AHC)
- HDBSCAN
- KMedoids

For fixed-`K` algorithms, clustering is evaluated for `K = 1` to `10`. HDBSCAN is evaluated separately because it determines the number of clusters automatically.

### 4. Evaluation

Clustering quality is measured using the **silhouette score**, which estimates how well samples fit within their assigned clusters compared with other clusters. The pipeline records scores across candidate configurations and selects the best-performing result automatically.

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
python clustering.py data.txt
```

Replace `data.txt` with your input dataset file.

Main outputs:

- `label.txt` - final cluster assignment for each input sentence, one label per line
- `silhouette_plot.png` - line plot comparing silhouette scores across clustering methods and `K` values
- `clustering_results.csv` - detailed output containing each sentence, its assigned cluster, and the selected algorithm

Additional output:

- `silhouette_scores.csv` - tabular silhouette scores recorded during evaluation

## Results

The quality of clustering is assessed using silhouette score. Higher silhouette values indicate more coherent and better-separated clusters. The final selected model is the one that achieves the highest valid silhouette score among the evaluated algorithms and parameter settings.

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

Core libraries used in this project include:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `sentence-transformers`
- `torch`
- `transformers`
- `umap-learn`
- `hdbscan`
- `nltk`

Note: `KMedoids` requires `scikit-learn-extra` if that algorithm is to be enabled.

## Notes

- This project uses only local Python libraries and pretrained models
- No cloud APIs or external hosted ML services are used
- The pipeline is suitable for offline academic experimentation, subject to local model and package installation

## Author

Saif Ur Rehman
