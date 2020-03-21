'''
Created on May 19, 2014

@author: roberto

Some examples with the complete and simplified tag set used by NLTK

Ported to Python 3 and NLTK 3 on May 5, 2016
'''

import nltk

# prints the first 10 sentences of the Brown corpus, with their original tags, and
# wth the simplified tag set
print("---Regular Brown tagset:", nltk.corpus.brown.tagged_words()[1:10])

input("Press Enter to continue...")

# Extracts and prints the complete Brown estimated_tags (remove '' and modifiers '-NC', '-HL', '-TL' from tags)
# -NC=citations, -HL=word in headline, -TL=word in title
print("---Cleaned Brown tagset:")
brown_corpus_tags = sorted(set([t for (w,t) in nltk.corpus.brown.tagged_words()]))
cleaned_brown_corpus_tags =  [t for t in brown_corpus_tags if t != '' and "-NC" not in t and "-HL" not in t and "-TL" not in t]
print(cleaned_brown_corpus_tags)

input("Press Enter to continue...")

# Print help for each Brown tag
print("---Meaning of Brown corpus tags:")
for t in cleaned_brown_corpus_tags:
    print("\tTAG:", t)
    print("\tDescription:\n", nltk.help.brown_tagset(t), "\n")

input("Press Enter to continue...")

# Print most ambiguous words (i.e., words with several POS's) in the 'news' section 
# of the Brown corpus
print("---Most ambiguous words:")
brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news')
# data = p(tag | word)
data = nltk.ConditionalFreqDist([(word.lower(), tag) for (word, tag) in brown_news_tagged])
# E.g., data['open'] = {'ADJ': 13, 'V': 11, 'N': 8, 'ADV': 1}, i.e., p(tag | 'open')
#       data['open'].keys() = ['ADJ', 'V', 'N', 'ADV']
for word in data.conditions():  # i.e., the set of words
    if (len(data[word]) > 3):  # print words with more than 3 tags
        estimated_tags = data[word].keys()
        print(word, "-",' '.join(estimated_tags))
