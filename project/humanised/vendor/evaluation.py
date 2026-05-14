from util import *
import math
# Add your import statements here


class Evaluation():

	def _get_true_doc_ids(self, qrels, query_id):
		query_id = str(query_id)
		return [item["id"] for item in qrels if str(item["query_num"]) == query_id]
	def _get_true_doc_ids_with_position(self, qrels, query_id):
		query_id = str(query_id)

		l = [(item["id"], item["position"]) for item in qrels if str(item["query_num"]) == query_id]
		l.sort(key=lambda x: x[1])
		return l
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

		precision = -1
		
		retrieved = set(query_doc_IDs_ordered[:k])
		retrieved = set([str(doc_id) for doc_id in retrieved])
		relevant = set(true_doc_IDs).intersection(retrieved)
		if len(retrieved) == 0:
			return 0.0
		precision = len(relevant) / len(retrieved)
		
		return precision

	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanPrecision = -1

		total_precision = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_precision += self.queryPrecision(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		meanPrecision = total_precision / float(num_queries)


		return meanPrecision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		recall = -1
		
		retrieved = set(query_doc_IDs_ordered[:k])
		retrieved = set([str(doc_id) for doc_id in retrieved])
		relevant = set(true_doc_IDs).intersection(retrieved)
		if len(retrieved) == 0:
			return 0.0
		recall = len(relevant) / len(true_doc_IDs)
		
		#Fill in code here

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanRecall = -1

		#Fill in code here
		total_recall = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_recall += self.queryRecall(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		meanRecall = total_recall / float(num_queries)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		fscore = -1
		retrieved = set(query_doc_IDs_ordered[:k])
		retrieved = set([str(doc_id) for doc_id in retrieved])
		relevant = set(true_doc_IDs).intersection(retrieved)
		if len(retrieved) == 0:
			return 0.0
		recall = len(relevant) / len(true_doc_IDs)
		precision = len(relevant) / len(retrieved)
		if precision + recall == 0:
			return 0.0
		fscore = 2 * (precision * recall) / (precision + recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanFscore = -1

		#Fill in code here
		total_fscore = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids(qrels, query_id)
			total_fscore += self.queryFscore(doc_IDs_ordered[index], query_id, true_doc_IDs, k)
		meanFscore = total_fscore / float(num_queries)
		return meanFscore


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		nDCG = -1
		true_reli = {k:j for k,j in true_doc_IDs}
		query_doc_IDs_ordered = ([str(doc_id) for doc_id in query_doc_IDs_ordered])
		relevant_docs = set([(doc_id) for doc_id, _ in true_doc_IDs])
		def dcg(retrieved_docs):
			score = 0.0
			for rank, doc_id in enumerate(retrieved_docs, start=1):
				if doc_id in relevant_docs:
					score += (5-true_reli[doc_id]) / math.log2(rank + 1)
			return score

		ideal = sorted(true_doc_IDs, key=lambda x: x[1], reverse=False)[:k]
		ideal_dcg = dcg([doc_id for doc_id, _ in ideal])
		if ideal_dcg == 0:
			return 0.0
		d = dcg(query_doc_IDs_ordered[:k])
		nDCG = d / ideal_dcg
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanNDCG = -1

		#Fill in code here
		total_ndcg = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids_with_position(qrels, query_id)
			total_ndcg += self.queryNDCG(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		meanNDCG = total_ndcg / float(num_queries)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		avgPrecision = -1
		query_doc_IDs_ordered = ([str(doc_id) for doc_id in query_doc_IDs_ordered][:k])
		relevant_docs = set([(doc_id) for doc_id, _ in true_doc_IDs])
		num_relevant = 0
		sum_precision = 0.0
		for i in range(len(query_doc_IDs_ordered)):
			if query_doc_IDs_ordered[i] in relevant_docs:
				num_relevant += 1
				sum_precision += num_relevant / (i + 1)
		avgPrecision = sum_precision / len(relevant_docs) if len(relevant_docs) > 0 else 0.0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		meanAveragePrecision = -1
		
		total_map = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]
			true_doc_IDs = self._get_true_doc_ids_with_position(q_rels, query_id)
			total_map += self.queryAveragePrecision(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		meanAveragePrecision = total_map / float(num_queries)

		return meanAveragePrecision



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

		reciprocalRank = 0

		query_doc_IDs_ordered = ([str(doc_id) for doc_id in query_doc_IDs_ordered][:k])
		relevant_docs = set([(doc_id) for doc_id, _ in true_doc_IDs])
		for i, doc_id in enumerate(query_doc_IDs_ordered):
			if doc_id in relevant_docs:
				reciprocalRank = 1 / (i + 1)
				break

		return reciprocalRank


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

		meanReciprocalRank = -1
		
		total_rr = 0.0
		num_queries = min(len(doc_IDs_ordered), len(query_ids))
		for index in range(num_queries):
			query_id = query_ids[index]	
			true_doc_IDs = self._get_true_doc_ids_with_position(qrels, query_id)
			total_rr += self.queryReciprocalRank(doc_IDs_ordered[index], query_id, true_doc_IDs, k)

		meanReciprocalRank = total_rr / float(num_queries)

		return meanReciprocalRank
