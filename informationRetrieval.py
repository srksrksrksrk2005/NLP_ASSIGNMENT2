from util import *
import math
import numpy as np
# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.total_docs = None
		self.tfidf = None
		self.words = []
		self.docIDs = None
	
	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		index = {}
		for doc, docID in zip(docs, docIDs):
			for sentence in doc:
				for word in sentence:
					if word not in index:
						self.words.append(word)
						index[word] = {}
						index[word][docID] = 1
					else:
						if docID not in index[word]:
							index[word][docID] = 1
						else:
							index[word][docID] += 1
		self.index = index
		self.total_docs = len(docIDs)
		self.docIDs = docIDs
		for word in index:
			self.index[word]['idf'] = math.log(self.total_docs / len(self.index[word]))
		self.tfidf = np.zeros(shape = (self.total_docs, len(self.index)))
	
		for i in range(self.total_docs):
			for j in range(len(self.index)):
				# Use .get to handle words that do not occur in a particular document
				self.tfidf[i,j] = self.index[self.words[j]]['idf'] * self.index[self.words[j]].get(docIDs[i], 0)
		self.normalized_tfidf = self.tfidf / (1e-12 + np.linalg.norm(self.tfidf, axis = 1, keepdims = True))
		
	
	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		query_tfidf = np.zeros(shape = (len(queries), len(self.words)))
		for i in range(len(queries)):
			for sentence in queries[i]:
				for word in sentence:
					if word in self.index:
						query_tfidf[i, self.words.index(word)] += self.index[word]['idf']
		normalized_query_tfidf = query_tfidf / (1e-12 + np.linalg.norm(query_tfidf, axis = 1, keepdims = True))
		cosine_similarities = np.dot(normalized_query_tfidf, self.normalized_tfidf.T)
		
		for i in range(len(queries)):
			doc_IDs_ordered.append([self.docIDs[j] for j in list(np.argsort(cosine_similarities[i])[::-1])])

		return doc_IDs_ordered




