from util import *
import math
import numpy as np
from collections import defaultdict


class BM25Retrieval():
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b

        # populated by buildIndex
        self.index    = {}        # token → {docID: tf, 'df': int, 'idf': float}
        self.docIDs   = None
        self.doc_len  = {}        # docID → number of tokens
        self.avgdl    = 0.0
        self.N        = 0         # total number of documents
        self.vocab    = []        # ordered vocabulary list


    def buildIndex(self, docs, docIDs):
        index    = {}
        doc_len  = {}

        for doc, docID in zip(docs, docIDs):
            tokens = [tok for sent in doc for tok in sent]   # flatten
            doc_len[docID] = len(tokens)

            for token in tokens:
                if token not in index:
                    self.vocab.append(token)
                    index[token] = {}
                index[token][docID] = index[token].get(docID, 0) + 1

        self.index   = index
        self.docIDs  = docIDs
        self.doc_len = doc_len
        self.N       = len(docIDs)
        self.avgdl   = sum(doc_len.values()) / self.N if self.N else 1.0


        for token in index:
            df = len(index[token])  
            self.index[token]['idf'] = math.log(
                (self.N - df + 0.5) / (df + 0.5) + 1
            )

    def _score_doc(self, query_tokens, docID):
        score  = 0.0
        dl     = self.doc_len.get(docID, 0)
        norm   = 1 - self.b + self.b * (dl / self.avgdl)

        for token in query_tokens:
            if token not in self.index:
                continue
            tf  = self.index[token].get(docID, 0)
            idf = self.index[token]['idf']
            tf_bm25 = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            score  += idf * tf_bm25

        return score

    def rank(self, queries):
        doc_IDs_ordered = []

        for query in queries:
            query_tokens = [tok for sent in query for tok in sent]

            scores = {docID: self._score_doc(query_tokens, docID)
                      for docID in self.docIDs}

            ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
            doc_IDs_ordered.append(ranked)

        return doc_IDs_ordered
