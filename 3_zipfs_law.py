'''
@author: roberto

Check here for an explanation of the Zipf's Law:
http://nlp.stanford.edu/IR-book/html/htmledition/zipfs-law-modeling-the-distribution-of-terms-1.html

Ported to NLTK3 and Python 3 on May 23, 2016

'''

from nltk.corpus import gutenberg
from nltk import FreqDist
from matplotlib import pyplot

# Load a book and compute word distribution
words = gutenberg.words('austen-emma.txt')
fdist = FreqDist(words)

# create a list containing the counts of each word in the distribution
# fdist.most_common() considers items in decreasing order of frequency
count_list = [count for (word,count) in list(fdist.most_common()) if word.isalpha()]   

# plot the Zipf's Law with linear scales
pyplot.plot(list(range(len(count_list))), count_list)
pyplot.xlabel('rank')
pyplot.ylabel('freq')
pyplot.show()

input("Press Enter to continue...")

# plot the Zipf's Law with logarithmic scales
pyplot.plot(list(range(len(count_list))), count_list)
pyplot.xlabel('log_10(rank)')
pyplot.ylabel('log_10(freq)')
pyplot.xscale('log')
pyplot.yscale('log')
pyplot.show()


