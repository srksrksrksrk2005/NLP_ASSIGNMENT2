from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords
from collections import Counter



class StopwordRemoval():


	def fromList(self, text, stop_words=None):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		if stop_words is None:
			stop_words = set(stopwords.words('english'))

		for tokenizedSentence in text:
			sentence = []
			for word in tokenizedSentence:
				if word.lower() not in stop_words:
					sentence.append(word)
			stopwordRemovedText.append(sentence)


		return stopwordRemovedText

	def generateList(self, text, filterThreshold= 0.15, deltaCutoff=0.02, slope = False):
		tf = Counter()
		for Sentences in text:
			for word in Sentences:
				tf[word.lower()] += 1
		total_words = sum(tf.values())
		stop_words = set()
		if slope == False:	
			for word in tf:
				if tf[word] / total_words > filterThreshold:
					stop_words.add(word)
		else:
			tf_list = tf.most_common()
			delta = 0
			prev = tf_list[0][1]
			for word, count in tf_list:
				if prev - count > deltaCutoff * total_words:
					break
				if count / total_words > filterThreshold:
					stop_words.add(word)
				else:
					break
				prev = count
		
		return stop_words



	