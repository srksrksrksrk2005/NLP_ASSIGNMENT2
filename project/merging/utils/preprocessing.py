"""
Preprocessing Utilities
Shared preprocessing functions for text normalization and tokenization
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Handles text preprocessing: tokenization, stemming, stopword removal"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))
    
    def lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()
    
    def remove_special_chars(self, text):
        """Remove special characters and numbers"""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def segment_sentences(self, text):
        """Segment text into sentences"""
        return sent_tokenize(text)
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stopwords_set]
    
    def stem(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens):
        """Apply lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text, lowercase=True, remove_special=False, 
                        remove_sw=True, stem=False, lemmatize=False):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            lowercase: Apply lowercasing
            remove_special: Remove special characters
            remove_sw: Remove stopwords
            stem: Apply stemming
            lemmatize: Apply lemmatization
        
        Returns:
            List of preprocessed tokens
        """
        if lowercase:
            text = self.lowercase(text)
        
        if remove_special:
            text = self.remove_special_chars(text)
        
        tokens = self.tokenize(text)
        
        if remove_sw:
            tokens = self.remove_stopwords(tokens)
        
        if stem:
            tokens = self.stem(tokens)
        elif lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens
    
    def preprocess_batch(self, texts, **kwargs):
        """Preprocess a batch of texts"""
        return [self.preprocess_text(text, **kwargs) for text in texts]
    
    def get_token_string(self, tokens):
        """Convert token list to space-separated string"""
        if isinstance(tokens, list):
            return " ".join(tokens)
        return tokens


def preprocess_pipeline(text, config=None):
    """Quick preprocessing function"""
    processor = TextPreprocessor(config)
    return processor.preprocess_text(text)
