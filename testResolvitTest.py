import unittest
import json
from collections import OrderedDict
from nltk import pos_tag
from testResolvit import WordFrequencyAnalyzer

class Test(unittest.TestCase):
	
	def setUp(self):
		self.wfa = WordFrequencyAnalyzer()

	def test_remove_stopwords(self):
		result = self.wfa.removeStopwords(["be", "honest"])
		self.assertEqual( result, ["honest"])

	def test_remove_stopwords_From_Empty_List(self):
		result = self.wfa.removeStopwords([])
		self.assertEqual( result, [])

	def test_remove_punctuation(self):
		result = self.wfa.removePunct(["be?", "hones.t"])
		self.assertEqual( result, ["be", "honest"])

	def test_lemmatize_Text_From_IS_to_BE_And_Plural(self):
		tagged = pos_tag( "This is so nice dogs".split() ) #After removing punctuation
		result = self.wfa.lemmatizeText(tagged)
		self.assertEqual( result, ["this", "be", "so", "nice", "dog"])

	def test_lemmatize_Text_unique_words(self):
		tagged = pos_tag( "This can be a test".split() ) #After removing punctuation
		result = self.wfa.lemmatizeText(tagged)
		self.assertEqual( result, ["this", "can", "be", "a", "test"])

	def test_analyze_Text_And_Get_Stats_OK(self):
		text = "Dogs are wonderful. My dog is the best."
		resultList = [
		{
			"word": "best",
			"total-occurances": 1,
			"sentence-indexes": "[1]"
		},
		{
			"word": "dog", #From Dogs and dog
			"total-occurances": 2,
			"sentence-indexes": "[0],[1]" 	
		},
		{
			"word": "my",
			"total-occurances": 1,
			"sentence-indexes": "[1]"			
		},
		{	
			"word": "wonderful",
			"total-occurances": 1, 
			"sentence-indexes": "[0]"
		}]
		ordList = [OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True)) for d in resultList]
		jsonOut = {"results": ordList }   # "The" word is removed
		result = self.wfa.analyzeTextAndGetStats(text)
		self.assertEqual(result, json.dumps(jsonOut, indent=4))


if __name__ == '__main__':
    unittest.main()
