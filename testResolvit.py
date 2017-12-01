#!usr/bin/python
from sets import Set
import json
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

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


	def posTagger(self, sentence):
		return pos_tag(sentence)

	def removeStopwordsAndPunct(self, tokens ):
		cleanTokens = [''.join( ch for ch in word if ch not in ".,\":" ) for word in tokens]
		finalTokens = [ token for token in cleanTokens if token.lower() not in self.stopwords ]
		return finalTokens

	def lemmatizeText(self, tokens ):
		result = []
		for word, tag in tokens:
			tlTag = self.translateTag(tag)
			if tlTag is None:
				result.append(self.lemmatizer.lemmatize(word))
			else:
				result.append(self.lemmatizer.lemmatize(word, pos=tlTag))
		return result

	def analyzeTextAndGetStats(self, text ):
		sentenceTable =[]
		index = 0
		for sentence in sent_tokenize( text ):
			tagged_sent = self.posTagger( sentence.split() )
			tokens =  self.lemmatizeText( tagged_sent )
			finalTokens = self.removeStopwordsAndPunct(tokens)
			sentenceTable = sentenceTable + [ (index, word) for word in finalTokens ]
			index = index + 1

		res = {}
		for sentenceIdx , word in sentenceTable:
			t = res.get( word )
			if not t:
				t = ( 0, set() )
			count, setIdx = t
			setIdx.add(sentenceIdx) 
			newElem = ( count + 1 , setIdx )
			res.update( [ ( word , newElem  ) ] )
	
		fin = map(lambda elem: { "word" : elem[0], "total-ocurrences" : elem[1][0], "sentence-indexes" : list(elem[1][1]) } , res.iteritems() )
		return json.dumps({ "results" : fin }, sort_keys=True, indent=4)




test = "Take this paragraph of text and return an alphabetized list of ALL unique words.  A unique word is any form of a word often communicated with essentially the same meaning. For example, fish and fishes could be defined as a unique word by using their stem fish. For each unique word found in this entire paragraph, determine the how many times the word appears in total. Also, provide an analysis of what sentence index position or positions the word is found. The following words should not be included in your analysis or result set: \"a\", \"the\", \"and\", \"of\", \"in\", \"be\", \"also\" and \"as\".  Your final result MUST be displayed in a readable console output in the same format as the JSON sample object shown below."

wordAnalyzer = WordFrequencyAnalyzer()
result = wordAnalyzer.analyzeTextAndGetStats(test)
print(result)