from util import *
import math
import numpy as np
from collections import defaultdict


class NgramRetrieval():
    # uses unigrams and bigrams (unless n specified manually)
    # build combined TF-IDF index and rank using cosine
    
    def __init__(self, n=2):
        # n is the maximum n-gram size 
        
        self.n = n
        self.index = None
        self.total_docs = None
        self.tfidf = None
        self.vocab = []
        self.docIDs = None
        self.normalized_tfidf = None

    def _extract_ngrams(self, sentences):
        
        tokens = []
        for sentence in sentences:
            words = sentence

            # unigrams
            tokens.extend(words)

            # higher-order n-grams
            for k in range(2, self.n + 1):
                for i in range(len(words) - k + 1):
                    ngram = "_".join(words[i:i + k])
                    tokens.append(ngram)

        return tokens

    def buildIndex(self, docs, docIDs):
        # build an inverted index and TF-IDF matrix over unigrams + bigrams.
        
        index = {}

        for doc, docID in zip(docs, docIDs):
            
            # get all n-gram tokens for this document
            tokens = self._extract_ngrams(doc)

            for token in tokens:
                if token not in index:
                    self.vocab.append(token)
                    index[token] = {}
                    index[token][docID] = 1
                else:
                    if docID not in index[token]:
                        index[token][docID] = 1
                    else:
                        index[token][docID] += 1

        self.index = index
        self.total_docs = len(docIDs)
        self.docIDs = docIDs

        # idf
        for token in index:
            df = len([k for k in index[token] if k != 'idf'])
            self.index[token]['idf'] = math.log(self.total_docs / (df + 1e-12))

        # TF-IDF matrix  (docs x vocab)
        self.tfidf = np.zeros((self.total_docs, len(self.vocab)))
        for i, docID in enumerate(docIDs):
            for j, token in enumerate(self.vocab):
                tf = self.index[token].get(docID, 0)
                self.tfidf[i, j] = self.index[token]['idf'] * tf

        # L2-normalise rows
        norms = np.linalg.norm(self.tfidf, axis=1, keepdims=True)
        self.normalized_tfidf = self.tfidf / (norms + 1e-12)

    def rank(self, queries):
        #cosine similarity based
        
        doc_IDs_ordered = []

        query_tfidf = np.zeros((len(queries), len(self.vocab)))
        vocab_index = {token: idx for idx, token in enumerate(self.vocab)}

        for i, query in enumerate(queries):
            tokens = self._extract_ngrams(query)
            for token in tokens:
                if token in vocab_index:
                    query_tfidf[i, vocab_index[token]] += self.index[token]['idf']

        # L2-normalise
        q_norms = np.linalg.norm(query_tfidf, axis=1, keepdims=True)
        normalized_query_tfidf = query_tfidf / (q_norms + 1e-12)

        cosine_similarities = np.dot(normalized_query_tfidf, self.normalized_tfidf.T)

        for i in range(len(queries)):
            ranked_indices = np.argsort(cosine_similarities[i])[::-1]
            doc_IDs_ordered.append([self.docIDs[j] for j in ranked_indices])

        return doc_IDs_ordered
