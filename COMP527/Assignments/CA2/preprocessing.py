# preprocessing.py

from sklearn.feature_extraction.text import TfidfVectorizer
file_path = '/home/saif/Projects/UoL/COMP527/Assignments/CA2/data_train.txt'
def load_data(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                sentences.append(parts[1])
    return sentences


def preprocess(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return X