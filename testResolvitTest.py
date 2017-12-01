import unittest
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
		self.assertEqual( result, ["This", "be", "so", "nice", "dog"])


if __name__ == '__main__':
    unittest.main()