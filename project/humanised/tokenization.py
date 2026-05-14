import re

import spacy
from nltk.tokenize import TreebankWordTokenizer
from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)


class Tokenization():

	def __init__(self):

		self.nlp = spacy.load("en_core_web_sm")
	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		for sentence in text:
			tokenizedSentence  = re.findall(r'\b\w+\b', sentence)
			tokenizedText.append(tokenizedSentence)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		tokenizer = TreebankWordTokenizer()
		for sentence in text:
			tokenizedSentence = tokenizer.tokenize(sentence)
			tokenizedText.append(tokenizedSentence)

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		for doc in self.nlp.pipe(text):
			tokenisedSentence = [token.text for token in doc]
			tokenizedText.append(tokenisedSentence)

		return tokenizedText
