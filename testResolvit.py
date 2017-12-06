#!usr/bin/python
from sets import Set
from collections import OrderedDict
import json
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

#Should check if it's already installed
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class WordFrequencyAnalyzer():

	def __init__(self):
		self.stopwords = ["a", "the", "and", "of", "in", "be", "also" , "as"]
		self.lemmatizer = WordNetLemmatizer()

	def translateTag(self, tag):
		if tag.startswith('J'):
			return wn.ADJ
		elif tag.startswith('V'):
			return wn.VERB 
		elif tag.startswith('N'):
			return wn.NOUN
		elif tag.startswith('R'):
			return wn.ADV
		else:
			return None

	def removePunct(self, tokens):
		return [''.join( ch for ch in word if ch not in ".,\":?" ) for word in tokens]


	def removeStopwords(self, tokens ):
		return [ token.encode("ascii", "ignore") for token in tokens if token.lower() not in self.stopwords ]
		

	def lemmatizeText(self, tokens ):
		result = []
		for word, tag in tokens:
			tlTag = self.translateTag(tag)
			lemmaWord = ""
			if tlTag is None:
				lemmaWord = self.lemmatizer.lemmatize(word.lower())
			else:
				lemmaWord = self.lemmatizer.lemmatize(word.lower(), pos=tlTag)
			if word.isupper():
				lemmaWord = lemmaWord.upper()
			result.append(lemmaWord)

		return result

	def buildWordDict(self, sentenceTable):
		result = {}

		for sentenceIdx , word in sentenceTable:
			t = result.get( word )
			if not t:
				t = ( 0, set() )
			count, setIdx = t
			setIdx.add(sentenceIdx) 
			countAndIdxUpdated = ( count + 1 , setIdx )
			result.update( [ ( word , countAndIdxUpdated ) ] )

		return result

	def formatIdx(self, idxSet):
		output = ""
		for i in idxSet:
			output = output + "[" + str(i) +"],"
		return output.rstrip(',')

	def analyzeTextAndGetStats(self, text ):
		sentenceTable =[]
		index = 0

		for sentence in sent_tokenize( text ):
			tagged_sent = pos_tag( self.removePunct(sentence.split()) )
			tokens =  self.lemmatizeText( tagged_sent )
			finalTokens = self.removeStopwords(tokens)
			sentenceTable = sentenceTable + [ (index, word) for word in finalTokens ]
			index = index + 1

		wordDict = self.buildWordDict(sentenceTable)
		ordered = OrderedDict(sorted(wordDict.items(), key=lambda t: t[0].lower()))
		# e[0] is word, e[1] is tuple (count, setOfSentenceIndexes )
		fin = map(lambda e: OrderedDict(
							sorted({ 'word': e[0],
									 'sentence-indexes': self.formatIdx(e[1][1]),
									 'total-occurances': e[1][0] }.iteritems(),
									  key=lambda t:t[0], reverse=True)),
									  ordered.iteritems() )
		return json.dumps({ "results" : fin }, indent=4 )


