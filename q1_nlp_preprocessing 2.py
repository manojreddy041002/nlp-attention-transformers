"""
Part II - Q1: NLP Preprocessing Pipeline
CS5720 Neural Network and Deep Learning - Home Assignment 4

Steps: tokenize -> remove stopwords -> stem (Porter stemmer).
Requires: pip install nltk
"""
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the NLTK resources once (no-op if already present)
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)


def preprocess(sentence):
    """Tokenize, remove stopwords/punctuation, and stem a sentence.
    Prints the result of each stage and returns all three lists."""
    # 1. Tokenization: split the sentence into words and punctuation marks
    tokens = word_tokenize(sentence)

    # 2. Stopword removal: drop common English words (compare in lowercase)
    #    and punctuation tokens so only meaningful words remain
    stop_words = set(stopwords.words("english"))
    filtered = [t for t in tokens
                if t.lower() not in stop_words and t not in string.punctuation]

    # 3. Stemming: reduce each remaining word to its root form
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in filtered]

    print("Original Tokens:          ", tokens)
    print("Tokens Without Stopwords: ", filtered)
    print("Stemmed Words:            ", stemmed)
    return tokens, filtered, stemmed


if __name__ == "__main__":
    text = "NLP techniques are used in virtual assistants like Alexa and Siri."
    preprocess(text)
