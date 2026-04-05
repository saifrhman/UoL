# preprocessing.py
"""
Text preprocessing for clustering assignment.

Usage:
    python3 preprocessing.py data_train.txt

This script:
1. Loads raw texts from a file
2. Cleans and normalizes the text
3. Saves preprocessed texts to preprocessed_texts.txt
"""

from __future__ import annotations

import re
import string
import sys
from pathlib import Path
from typing import List

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("Error: nltk is required. Install with: pip install nltk")
    sys.exit(1)


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if missing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean a single text document.

    Steps:
    - lowercase
    - remove HTML tags
    - remove URLs
    - remove emails
    - remove non-ASCII characters
    - remove punctuation
    - remove numbers
    - tokenize
    - remove stopwords
    - lemmatize
    """
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    cleaned_tokens = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in STOP_WORDS:
            continue
        lemma = LEMMATIZER.lemmatize(token)
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def load_texts(file_path: str) -> List[str]:
    """Load texts line-by-line from input file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        texts = [line.strip() for line in f if line.strip()]

    return texts


def save_texts(texts: List[str], output_path: str) -> None:
    """Save preprocessed texts line-by-line."""
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 preprocessing.py data_train.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "preprocessed_texts.txt"

    texts = load_texts(input_file)
    cleaned_texts = [clean_text(text) for text in texts]

    save_texts(cleaned_texts, output_file)

    print(f"Loaded {len(texts)} texts")
    print(f"Saved preprocessed texts to {output_file}")


if __name__ == "__main__":
    main()