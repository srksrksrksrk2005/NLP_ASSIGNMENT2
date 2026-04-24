from util import *
import math
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt',     quiet=True)


class WordNetRetrieval():
    """
    strategy: 
    1. for each content word w in a sentence, collect its WordNet synsets
    2. for each synset, count how many words in the synset's definition+examples
       also appear in the surrounding sentence
    3. pick the one with the highest overlap
    """

    def __init__(self, max_synonyms=3, context_window=10):
        self.max_synonyms   = max_synonyms
        self.context_window = context_window

        self.index              = None
        self.total_docs         = None
        self.tfidf              = None
        self.vocab              = []       # ordered list of index tokens
        self.docIDs             = None
        self.normalized_tfidf   = None


    def _lesk_synset(self, word, context_words, pos=None):
        # simplified Lesk algorithm
        # return the best synset for `word`
        
        synsets = wordnet.synsets(word, pos=pos)
        if not synsets:
            return None

        context_set = set(context_words)
        best_synset  = synsets[0]   # default: most common sense
        best_score   = -1

        for synset in synsets:
            # definition + examples
            
            signature = set(
                word_tokenize(synset.definition().lower())
            )
            for example in synset.examples():
                signature.update(word_tokenize(example.lower()))

            score = len(signature & context_set)
            if score > best_score:
                best_score  = score
                best_synset = synset

        return best_synset

    def _expand_token(self, word, sentence_words):
        #context excludes the word itself
        context = [w for w in sentence_words if w != word]

        # pos-tag the word in context for better lookup
        tagged = nltk.pos_tag([word])
        pos    = get_wordnet_pos(tagged[0][1]) if tagged else None

        synset = self._lesk_synset(word, context, pos=pos)
        if synset is None:
            return [word]

        # Collect lemma names, excluding the word itself and multi-word lemmas
        synonyms = []
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace('_', ' ').lower()
            if lemma_name != word and ' ' not in lemma_name:
                synonyms.append(lemma_name)
            if len(synonyms) >= self.max_synonyms:
                break

        return [word] + synonyms

    def _expand_sentences(self, sentences):
        all_tokens = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                # Local context window around the current word
                start   = max(0, i - self.context_window)
                end     = min(len(sentence), i + self.context_window + 1)
                context = sentence[start:end]

                expanded = self._expand_token(word, context)
                all_tokens.extend(expanded)
        return all_tokens


    def buildIndex(self, docs, docIDs):
        index = {}

        for doc, docID in zip(docs, docIDs):
            tokens = self._expand_sentences(doc)

            for token in tokens:
                if token not in index:
                    self.vocab.append(token)
                    index[token] = {docID: 1}
                else:
                    if docID not in index[token]:
                        index[token][docID] = 1
                    else:
                        index[token][docID] += 1

        self.index      = index
        self.total_docs = len(docIDs)
        self.docIDs     = docIDs

        # IDF
        for token in index:
            df = len([k for k in index[token] if k != 'idf'])
            self.index[token]['idf'] = math.log(self.total_docs / (df + 1e-12))

        # TF-IDF matrix
        self.tfidf = np.zeros((self.total_docs, len(self.vocab)))
        for i, docID in enumerate(docIDs):
            for j, token in enumerate(self.vocab):
                tf = self.index[token].get(docID, 0)
                self.tfidf[i, j] = self.index[token]['idf'] * tf

        # L2-normalise
        norms = np.linalg.norm(self.tfidf, axis=1, keepdims=True)
        self.normalized_tfidf = self.tfidf / (norms + 1e-12)

    def rank(self, queries):
        doc_IDs_ordered = []
        vocab_index     = {token: idx for idx, token in enumerate(self.vocab)}

        query_tfidf = np.zeros((len(queries), len(self.vocab)))

        for i, query in enumerate(queries):
            tokens = self._expand_sentences(query)
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
