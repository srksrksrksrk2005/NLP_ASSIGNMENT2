import nltk

from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class InflectionReduction:

	def __init__(self):
		self.stemmer = None
		self.lemmatizer = None

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = []
		if not self.stemmer:
			self.stemmer = PorterStemmer()

		for tokenizedSentence in text:
			reducedText.append([self.stemmer.stem(token) for token in tokenizedSentence])
			
		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = []

		if not self.lemmatizer:
			self.lemmatizer = WordNetLemmatizer()

		for tokenizedSentence in text:
			pos_tags = nltk.pos_tag(tokenizedSentence)
			pos_tags = [(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
			reducedText.append([self.lemmatizer.lemmatize(token, pos) for token, pos in pos_tags])

		return reducedText



	def reduce(self, text, mode='stem'):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		if mode == 'stem':
			reducedText = self.porterStemmer(text)
		elif mode == 'lemmatize':
			reducedText = self.wordnetLemmatizer(text)

		return reducedText
