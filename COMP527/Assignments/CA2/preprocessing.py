from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


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
    )


def preprocess(sentences: list[str]):
    """Vectorize the input text into a sparse TF-IDF feature matrix."""
    vectorizer = build_vectorizer()
    return vectorizer.fit_transform(sentences)
