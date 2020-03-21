'''
Created on May 19, 2014

@author: roberto

A simple HMM POS tagger, trained on the Brown corpus; "closed vocabulary" assumption

Ported to Python 3 and NLTK 3 on May 5, 2016
'''

import re                         # for regular expressions
from nltk.corpus import brown
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.probability import SimpleGoodTuringProbDist
from nltk import ConfusionMatrix


def load_corpus():
    """Load tagged corpus, and clean words and tags"""
    # the tagged corpus, divided by sentences
    # e.g., [[('the','DET-NC'),('equation','N-NC'),...,('.','.')],[("in","P-TL"),("addition","N-TL"),...,('.','.')],...]
    fully_tagged_sentences = brown.tagged_sents(categories='news')

    tagged_sentences = []
    tag_set = set()
    word_types = set()

    for tagged_sentence in fully_tagged_sentences:
        #remove modifiers '-NC', '-HL', '-TL' from tags
        tagged_sentence = [(word.lower(),re.sub("-NC|-HL|-TL","",tag)) for (word,tag) in tagged_sentence]
        tagged_sentences += [tagged_sentence]  # add tagged sentence to the list

        # add each word and the corresponding tag to the respective sets
        for word,tag in tagged_sentence:
            word_types.add(word)   # it's a set: do not add duplicates
            tag_set.add(tag)       # it's a set: do not add duplicates

    return tagged_sentences, word_types, tag_set


def split_corpus(tagged_sentences, train_set_fraction):
    """Prepare train set and test set"""    
    train_set_limit = int(train_set_fraction * len(tagged_sentences))
    train_set = tagged_sentences[:train_set_limit]
    test_set =  tagged_sentences[train_set_limit:]
    return train_set, test_set


def train(train_set, word_types, tag_set):
    """
    Training...
    Called this way, the HMM knows the whole set of tags and the whole set of words (no "unknown" word and/or tag during test)
    """
    trainer = HiddenMarkovModelTrainer(list(tag_set), list(word_types)) # tag_set and word_types are sets: I need to create lists
    # GoodTuring smoothing
    # see: https://nltk.googlecode.com/svn/trunk/doc/api/nltk.probability.SimpleGoodTuringProbDist-class.html
    #      http://en.wikipedia.org/wiki/Additive_smoothing
    hmm = trainer.train_supervised(train_set, estimator=lambda fd, bins: SimpleGoodTuringProbDist(fd, bins))
    return hmm


def test(hmm, test_set):
    """testing with a list of tagged sentences..."""
    hmm.test(test_set, verbose=False)
    print()


def example(hmm, test_set, n1, n2):
    """Try to tag sentences between n1 and n2 (excluded) of the test set; just to show the result..."""
    estimated_tags = []
    gold_tags = []
    for test_sentence in test_set[n1:n2]:

        # the zip() function with the "*" operator can be used to unzip the list
        # see: https://stackoverflow.com/questions/7558908/unpacking-a-list-tuple-of-pairs-into-two-lists-tuples
        # [("this","is")]         [("PP","VB")]    <-- zip(*(["this","PP"],["is","VB"]))
        unlabelled_test_sentence, test_sentence_tags = zip(*test_sentence)

        # decoding...
        test_sentence_estimated_tags = hmm.best_path(unlabelled_test_sentence)

        # [("this","PP"),("is","VB")] --> "this/PP is/VB"
        print("Test: %s" % ' '.join([word+"/"+tag for (word,tag) in test_sentence]))

        # e.g.: zip(["this","is"],["PP","VB"]) ---> [("this","PP"),("is","VB")]
        print("HMM : %s" %  ' '.join([word+"/"+tag for
                                      (word,tag) in zip(unlabelled_test_sentence, test_sentence_estimated_tags)]))

        # e.g.: zip(['PP', 'NN', 'VB'],['PP', 'NN', 'NN']) --> [('PP','PP'),('NN','NN'),('VB','NN')]
        comparation_list = [1 if tag1==tag2 else 0 for
                            (tag1,tag2) in zip(test_sentence_tags, test_sentence_estimated_tags)]  # e.g.: --> [1, 1, 0]

        print("Comparation:", comparation_list)
        print("Accuracy   : %.2f\n" % (sum(comparation_list) / len(test_sentence) * 100))  # --> sum([1, 1, 0]) / 3 = 2/3

        estimated_tags += test_sentence_estimated_tags  # collects estimated tags, for further use
        gold_tags += test_sentence_tags                 # collects correct tags, for further use

    # prints confusion matrix
    print(ConfusionMatrix(gold_tags,estimated_tags))


def main():
    train_set_fraction = 0.8  # 80 %
    n1 = 0        # for the example: tag sentences i: n1 <= i < n2
    n2 = 10
    
    tagged_sentences, words, tag_set = load_corpus()
    train_set, test_set = split_corpus(tagged_sentences, train_set_fraction)
    hmm = train(train_set, words, tag_set)
    test(hmm, test_set)
    example(hmm, test_set, n1, n2)


if __name__ == '__main__':
    main()