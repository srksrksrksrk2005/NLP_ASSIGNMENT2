import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "into", "about", "than", "so", "such", "can",
    "could", "would", "should", "do", "does", "did", "done", "have", "has", "had", "we", "you",
    "they", "he", "she", "them", "his", "her", "our", "your", "their",
}


class Preprocessor:
    """A lightweight preprocessing pipeline for Cranfield experiments."""

    def __init__(self, use_lemmatization: bool = True) -> None:
        self.use_lemmatization = use_lemmatization
        self._ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._load_stopwords()

    @staticmethod
    def _ensure_nltk_resources() -> None:
        resources = [
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for resource_path, package_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package_name, quiet=True)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]

    @staticmethod
    def _tokenize(sentence: str) -> List[str]:
        # Keep alphanumeric tokens and simple hyphenated forms.
        return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?", sentence.lower())

    def _normalize_token(self, token: str) -> str:
        if not self.use_lemmatization:
            return token
        return self.lemmatizer.lemmatize(token)

    def _load_stopwords(self) -> set:
        try:
            return set(stopwords.words("english"))
        except LookupError:
            return FALLBACK_STOPWORDS

    def preprocess_text(self, text: str) -> List[List[str]]:
        processed_sentences: List[List[str]] = []
        for sentence in self._split_sentences(text):
            tokens = [self._normalize_token(t) for t in self._tokenize(sentence)]
            cleaned = [t for t in tokens if t and t not in self.stop_words]
            if cleaned:
                processed_sentences.append(cleaned)
        return processed_sentences

    def preprocess_corpus(self, texts: List[str]) -> List[List[List[str]]]:
        return [self.preprocess_text(text) for text in texts]
