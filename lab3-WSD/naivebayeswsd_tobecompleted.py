'''
Created on May 13, 2014

@author: roberto

Trains and Tests a Naive Bayes classifier for recognizing "apple" as a fruit or as a company

Texts from: http://www.litfuel.net/plush/?postid=200
See the NLTK book: http://www.nltk.org/book/ch06.html

Ported to Python 3 on May 5, 2016

NB: this is a simplified classifier, as:
- It only classifies one word
- lemmas are used for both co-occurence and collocation
- The testing procedure does not apply the cross-validation methodology
'''

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk import WordNetLemmatizer
from nltk.probability import FreqDist  # this is useful, somewhere...
import random


def extract_context(list_of_lemmas, idx, context_limit):
	''' 
	Extracts the left and right contexts (i.e., lists of lemmas) of the lemma in position 'idx'
	- list_of_lemmas: list containing a lemmatized text;
	                  e.g.: ['jenn_barthole', 'apple', 'survey', 'are', 'apple', 'product', 'good', ...]
	- idx: the position of the target lemma inside the list of lemmas
	- context_limit: how many lemmas to consider, before and after idx
	- returns:
		 - the list of lemmas, with length context_limit, before idx;
		   complete with '' if no enough lemmas are available before idx
		 - the list of lemmas, with length context_limit, after idx;
		   complete with '' if no enough lemmas are available after idx
	'''
	# COMPLETE HERE
	# NB: adds '' to fill left and right contexts, if they are close to the beginning or the end of the text
	if (idx - context_limit) >= 0:
		left_context = list_of_lemmas[idx-context_limit:idx]

	else :
		if i <= context_limit - idx :
			num_zero[ i ] += [ ' ' ]
		left_context = num_zero[ i ] + [ x[ 0 : idx - 1 ] for x in list_of_lemmas]

	if ((len( list_of_lemmas ) - idx) - context_limit) >= 0 :
		right_context = [ x[ idx - 1 : idx + context_limit ] for x in list_of_lemmas ]
	else :
		if i <= context_limit - (len( list_of_lemmas ) - idx) :
			num_zero[ i ] += [ ' ' ]
		right_context = [ x[ idx - 1 : end ] for x in list_of_lemmas ] + num_zero[ i ]

	return left_context, right_context
	# E.g. ['jenn_barthole']   and   ['survey']


def get_best_co_occurring_lemmas(target_lemma, n, context_limit, texts):
	'''
	Returns the n most frequently co-occurring lemmas of the given target lemma
	- target_lemma: the lemma for which the co-occurring lemma must be found
	- n: how many co-occurring lemmas to retain
	- context_limit: how many lemmas to consider, before and after the target lemma
	- texts: the corpus; an hashtable where items are file names;
		     e.g.: {'COMPANY': './data/apple-company-training.txt', 'FRUIT': './data/apple-fruit-training.txt'}
	- returns: a list of the n most co-occurring lemmas
	'''
	
	# COMPLETE HERE
	# - Accumulate all the co-occurring lemmas within the specified context limit
	# - Get the n most frequent co-occurring lemmas
	text = open(texts, 'r', encoding='utf-8').read()
	fdist = max(FreqDist(text))[:n]
	co_occurrences[0:n] = [w for w in text if FreqDist(w) == fdist]




   	return co_occurrences[:n]    # The first n lemmas
	# e.g., ['crisp', 'http', 'ipad', 'iphone', 'juice', 'like', 'mac', 'make', 'making', 'pie', 'product', 'sauce']


def features(target_word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas):
	'''
	Returns the feature set of a given target lemma: lemma, left collocations, right collocations, co_occurrence
	- target_word: the word to classify
	- left_context_lemmas: a list of lemmas, before the target word; e.g.: ['jenn_barthole']
	- right_context_lemmas: a list of lemmas, after the target word; e.g.: ['survey']
	- best_co_occurring_lemmas: the list of best co-occurring lemmas, for the target word;
	  e.g.: e.g., ['crisp', 'http', 'ipad', 'iphone', 'juice', 'like', 'mac', 'make', 'making', 'pie', 'product', 'sauce']
	- returns: an hashtable {'word': ..., 'left_collocations': ..., 'right_collocations': ..., 'co_occurrence': ...}
	'''

	# COMPLETE HERE
	# - Build co-occurrence vector, e.g. [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	# - returns the dictionary; notice that list are mutable and cannot be used in a dict; you need to convert them to tuples

	return {'word': ..., 'left_collocations':..., 'right_collocations': ..., 'co_occurrence': ...}
	# e.g.: {'word': 'apples', 'left_collocations': ('jenn_barthole'), 'right_collocations': ('survey'),
	#	     'co_occurrence': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'word': 'apples'}

def add_lemmas(text):
	'''	
	Lemmatizes the text, using WordNet; returns (lemma, word) for each word in text
	NB: discards stopwords and tokens of 1 character	
	- text: list of tokens of the document to lemmatize
	- returns: a list of pairs (lemma, word)
	'''
	tokenizer = WordPunctTokenizer()
	tokens = tokenizer.tokenize(text) # notice that the tokenizer splits words like '#apple' into '#', 'apple'		
	lem = WordNetLemmatizer()
	result =  [(lem.lemmatize(x.lower()),x.lower()) for x in tokens	if x not in stopwords.words('english') and len(x) > 1]
	return result # the pairs (lemma, word) 


def main():
	# Define constants
	train_test_texts = {}  # a void hash table
	train_test_texts['FRUIT'] = './data/apple-fruit-training.txt'
	train_test_texts['COMPANY'] = './data/apple-company-training.txt'
	experiment_text = './data/apple-tweets.txt'
	lemma_to_classify = 'apple'   # Give it in its base form, so 'apple' and 'apples' are both classified
	context_limit = 1   # Context will be +/- 1
	n_for_co_occurring = 12 # How many co-occurring lemmas to retain; e.g. retain the 12 best co-occurring lemmas
	train_set_fraction = 0.8  # 80 %

	# Build vector of the best co-occurring lemmas, for all the meaning of the lemma to classify
	best_co_occurring_lemmas = get_best_co_occurring_lemmas(lemma_to_classify, n_for_co_occurring, \
	                                                        context_limit, train_test_texts)

	# Loop through each item, grab the text, tokenize it and create a training feature with it
	featuresets = []
	for sense, training_file in iter(train_test_texts.items()):  # iter() creates couples (index, item); index is a string here
		print("Training %s..." % sense)
		text = open(training_file, 'r', encoding='utf-8').read()		
		list_of_lemmas_words = add_lemmas(text)
		list_of_lemmas = [x[0] for x in list_of_lemmas_words]	# Retains only lemmas
		for idx, (lemma,word) in enumerate(list_of_lemmas_words): # After enumerate():[(1,('cats','cat')),(2,('dog','dog')),...]
			if lemma == lemma_to_classify:
				left_context_lemmas, right_context_lemmas = extract_context(list_of_lemmas, idx, context_limit)		
				# Append a new tuple to the list
				# Notice we use word, not lemma as a feature
				featuresets += [(features(word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas), sense)]

		# COMPLETE HERE: print how many samples of lemma_to_classify, for the current sense

	# Select training set and test set
	# Shuffling is needed so that train_set and test_set will contain samples from both the first and the second file
	random.shuffle(featuresets)	 
	train_set_limit = int(train_set_fraction * len(featuresets))				
	train_set, test_set = featuresets[:train_set_limit], featuresets[train_set_limit:]
	
	# Train...
	classifier = NaiveBayesClassifier.train(train_set)
	
	# Test... Notice that each run will result in a different accuracy, as the train set is randomly chosen
	print("Accuracy:", accuracy(classifier, test_set))
	
	# Try to classify a new file and print the surrounding words of the classified lemma
	print("\nClassify new text: %s" % experiment_text)
	text = open(experiment_text, 'r', encoding='utf-8').read()
	list_of_lemmas_words = add_lemmas(text)
	list_of_lemmas = [x[0] for x in list_of_lemmas_words]	# Retains only lemmas of the current text
	for idx, (lemma,word) in enumerate(list_of_lemmas_words):   # After enumerate(): [(1,('cat','cats')),(2,('dog','dog')), ...]
		if lemma == lemma_to_classify:
			left_context_lemmas, right_context_lemmas = extract_context(list_of_lemmas, idx, context_limit)	
			decision = classifier.classify(features(word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas))

			# COMPLETE HERE
			# - Extract at most 10 tokens before the target lemma
			# - Extract at most 10 tokens after the target lemma
			print("Class: %s\tWhere(+-10): %s *%s* %s" % (decision, left_surrounding, lemma, right_surrounding))
			# E.g.: Class: COMPANY	Where(+-10): ['noosy', 'offers', 'hdmi', 'adapter', 'for', 'the', 'ipad', 'iphone',
		    #       'ipod', 'touch'] *apple* ['ipad', 'iphone', 'http', '://', 'bit', 'ly']

	# COMPLETE HERE
	# K-fold or random sampling validation
	# calculate:
	#   - average accuracy
	#   - standard deviation

if __name__ == '__main__':
	main()
	