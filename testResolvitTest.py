import unittest
import json
from nltk import pos_tag
from testResolvit import WordFrequencyAnalyzer

class Test(unittest.TestCase):
	
	def setUp(self):
		self.wfa = WordFrequencyAnalyzer()

	def test_remove_stopwords_And_Punct(self):
		result = self.wfa.removeStopwordsAndPunct(["be", "hones.t"])
		self.assertEqual( result, ["honest"])

	def test_remove_stopwords_And_Punct_From_Empty_List(self):
		result = self.wfa.removeStopwordsAndPunct([])
		self.assertEqual( result, [])

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
		jsonOut = {"results": [
		{	"sentence-indexes": [0],
			"total-ocurrences": 1, 
			"word": "wonderful"
		}, 
		{
			"sentence-indexes": [1],
			"total-ocurrences": 1,
			"word": "my"
		},
		{
			"sentence-indexes": [0, 1], 
			"total-ocurrences": 2, 
			"word": "dog" #From Dogs and dog
		}, 
		{
			"sentence-indexes": [1], 
			"total-ocurrences": 1, 
			"word": "best"
		}
		]}   # "The" word is removed

		result = self.wfa.analyzeTextAndGetStats(text)
		self.assertEqual(result, json.dumps(jsonOut, sort_keys=True, indent=4))


if __name__ == '__main__':
    unittest.main()