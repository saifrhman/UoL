from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))



nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text: str):
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]

def load_data(file_path: str) -> list[str]:
    """Load tab-separated text data and preserve the original row order."""
    sentences: list[str] = []

    with Path(file_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)

            # Skip the expected header row in the provided dataset format.
            if line_number == 0 and len(parts) == 2 and parts[0].strip().lower() == "id":
                continue

            if len(parts) == 2:
                sentences.append(parts[1].strip())
            else:
                sentences.append(parts[0].strip())

    return sentences


def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer suited to short and medium-length text."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        norm="l2",
        sublinear_tf=True,
        tokenizer=lemmatize_text,
    )


def preprocess(sentences: list[str]):
    """Vectorize the input text into a sparse TF-IDF feature matrix."""
    vectorizer = build_vectorizer()
    return vectorizer.fit_transform(sentences)


'''
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_data(file_path: str) -> List[str]:
    """
    Load tab-separated text data and preserve the original row order.
    Expected format:
    id \t text
    """
    sentences: List[str] = []

    with Path(file_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            # Skip header row if present
            if line_number == 0 and len(parts) == 2 and parts[0].strip().lower() == "id":
                continue

            if len(parts) == 2:
                sentences.append(parts[1].strip())
            else:
                sentences.append(parts[0].strip())

    return sentences


def build_embedding_model() -> SentenceTransformer:
    """
    Load a lightweight Sentence-BERT model.
    This model is fast and suitable for clustering tasks.
    """
    model_name = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


def preprocess(sentences: List[str]) -> np.ndarray:
    """
    Convert text into dense semantic embeddings using Sentence-BERT.

    Steps:
    - Encode sentences into embeddings
    - Normalize embeddings (important for cosine similarity)
    """
    if not sentences:
        raise ValueError("No sentences provided for preprocessing.")

    model = build_embedding_model()

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,  # VERY important for cosine similarity
        show_progress_bar=False
    )

    return embeddings
    '''