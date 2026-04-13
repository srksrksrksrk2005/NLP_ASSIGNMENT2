import math

# Add your import statements here


class Evaluation():

	def _normalize_id(self, value):
		return str(value)


	def _get_true_doc_ids(self, qrels, query_id):
		query_id = self._normalize_id(query_id)
		return [item["id"] for item in qrels if self._normalize_id(item["query_num"]) == query_id]

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		if k <= 0:
			return 0.0

		top_k = [self._normalize_id(doc_id) for doc_id in query_doc_IDs_ordered[:k]]
		if not top_k:
			return 0.0

		relevant_docs = {self._normalize_id(doc_id) for doc_id in true_doc_IDs}
		hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)

		return hits / float(len(top_k))


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		if not query_ids:
			return 0.0

		total_precision = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_precision += self.queryPrecision(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_precision / float(num_queries)

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		if k <= 0:
			return 0.0

		top_k = [self._normalize_id(doc_id) for doc_id in query_doc_IDs_ordered[:k]]
		relevant_docs = {self._normalize_id(doc_id) for doc_id in true_doc_IDs}
		if not relevant_docs:
			return 0.0

		hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)

		return hits / float(len(relevant_docs))


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		if not query_ids:
			return 0.0

		total_recall = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_recall += self.queryRecall(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_recall / float(num_queries)


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision + recall == 0:
			return 0.0

		return 2.0 * precision * recall / (precision + recall)


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		if not query_ids:
			return 0.0

		total_fscore = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_fscore += self.queryFscore(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_fscore / float(num_queries)
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		top_k = [self._normalize_id(doc_id) for doc_id in query_doc_IDs_ordered[:k]]
		relevant_docs = {self._normalize_id(doc_id) for doc_id in true_doc_IDs}
		if not relevant_docs:
			return 0.0

		def dcg(retrieved_docs):
			score = 0.0
			for rank, doc_id in enumerate(retrieved_docs, start=1):
				if doc_id in relevant_docs:
					score += 1.0 / math.log2(rank + 1)
			return score

		ideal_dcg = 0.0
		for rank in range(1, min(k, len(relevant_docs)) + 1):
			ideal_dcg += 1.0 / math.log2(rank + 1)

		if ideal_dcg == 0:
			return 0.0

		return dcg(top_k) / ideal_dcg


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		if not query_ids:
			return 0.0

		total_ndcg = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_ndcg += self.queryNDCG(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_ndcg / float(num_queries)


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		if k <= 0:
			return 0.0

		top_k = [self._normalize_id(doc_id) for doc_id in query_doc_IDs_ordered[:k]]
		relevant_docs = {self._normalize_id(doc_id) for doc_id in true_doc_IDs}
		if not relevant_docs:
			return 0.0

		hits = 0
		sum_precision = 0.0
		seen_relevant = set()
		for rank, doc_id in enumerate(top_k, start=1):
			if doc_id in relevant_docs and doc_id not in seen_relevant:
				hits += 1
				sum_precision += hits / float(rank)
				seen_relevant.add(doc_id)

		return sum_precision / float(len(relevant_docs))


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		if not query_ids:
			return 0.0

		total_average_precision = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(q_rels, query_id)
			total_average_precision += self.queryAveragePrecision(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_average_precision / float(num_queries)



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		top_k = [self._normalize_id(doc_id) for doc_id in query_doc_IDs_ordered[:k]]
		relevant_docs = {self._normalize_id(doc_id) for doc_id in true_doc_IDs}

		for rank, doc_id in enumerate(top_k, start=1):
			if doc_id in relevant_docs:
				return 1.0 / float(rank)

		return 0.0


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		if not query_ids:
			return 0.0

		total_reciprocal_rank = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_reciprocal_rank += self.queryReciprocalRank(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		return total_reciprocal_rank / float(num_queries)
