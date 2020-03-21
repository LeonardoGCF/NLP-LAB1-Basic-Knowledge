"""
Created on May 30, 2014
@author: roberto

A simple example of parsing with a CFG and a PCFG, both learned from the Penn treebank
Ported to Python3 on May 5, 2016
"""
from nltk.corpus import treebank
from nltk.grammar import Nonterminal, induce_pcfg
from nltk.grammar import CFG
from nltk.parse import EarleyChartParser
from nltk.parse.pchart import InsideChartParser


def generate_grammar_and_parsers(parsed_sents):
    # From sentences, extract the parsing tree and transform each tree to a list of CFG productions;
    # generate a set containing all the productions (without repetitions)
    tbank_productions_with_repet = [production for parsed_sent in parsed_sents for production in parsed_sent.productions()]
    tbank_productions = set(tbank_productions_with_repet)  # exclude repetitions
    print("Num. of unique productions read:", len(tbank_productions))

    # Build a CFG from the productions
    print("\nBuinding a CFG...")
    cfg_grammar = CFG(Nonterminal('S'), tbank_productions)  # a CFG
    print(cfg_grammar, end="\n\n")

    # CFG - An Earley parser
    cfg_earley_parser = EarleyChartParser(cfg_grammar, trace=3)
    # Build a PCFG from the productions

    print("Building a PCFG...")
    pcfg_grammar = induce_pcfg(Nonterminal('S'), tbank_productions_with_repet)  # a PCFG, here repetitions are needed!
    print(pcfg_grammar, end="\n\n")

    # Allocate a bottom-up chart parser for PCFG; see: http://www.nltk.org/_modules/nltk/parse/pchart.html
    pcfg_pchart_parser = InsideChartParser(pcfg_grammar)

    return cfg_earley_parser, pcfg_pchart_parser # return both parsers


def earley(parser, sentence, gold_tree):
    ''' Earley parser for CFG: generates a list of parse trees'''
    test_trees = parser.parse(sentence)  # get a list of Tree
    # very bad transforming a generator into a list, but I want to know the number of tree found
    # and it seems than it is not possible to scan a generator more than one time
    test_trees = list(test_trees)
    print("\n---> EARLEY CFG PARSER - TREES FOUND:", len(test_trees))
    # look for the correct tree
    for idx, test_tree in enumerate(test_trees):
        print("TREE: #%d" % idx)
        print(test_tree)
        if test_tree.productions() == gold_tree.productions():
            print("CORRECT TREE\n")
        else:
            print("WRONG TREE\n")


def pchart(parser, sentence, gold_tree):
    ''' Pchart parser for PCFG: generates a list of parse trees, with probabilities  '''

    test_trees = parser.parse(sentence) # get an iterator on a list of ProbabilisticTree
    # very bad transforming a list iterator into a list (it is needed by len(), as an iterator has no... length)
    test_trees = list(test_trees)
    print("\n---> PCHART PCFG PARSER - TREE(S) FOUND:", len(test_trees))

    # look for the best tree
    best_prob = 0.0
    for idx, test_tree in enumerate(test_trees):      # NB: enumerate([a,b,c]) --> ((1,a),(2,b),(3,c))
        print("\nTREE: #%d" % idx)
        print(test_tree)

        # if the probability of the current tree is higher than the current probability, it is the new "best tree"
        curr_prob = test_tree.prob()
        if curr_prob > best_prob:
            best_tree = test_tree
            best_prob = curr_prob

        # A parser does not have this information (it does not know the "correct tree"...)
        # This is just to understand whether the parser selected the correst tree, or a wrong tree, as its "best tree"
        if test_tree.productions() == gold_tree.productions():
            print("CORRECT TREE")
        else:
            print("WRONG TREE")

    # Return the most probable tree (the "best tree")
    # If the parser worked well, this tree is the CORRECT TREE   
    return best_tree   


def main():
    
    # Read sentence i of the treebank, with n1 <= i < n2
    n1 = 0   
    n2 = 10
    
    # Parse sentence j of the treebank, with m1 <= j < m2
    # NB: usually, one should _NOT_ test a parser on the same sentences used to train it!!!
    #     But, in this example, the grammar is so small that it is unlikely to parse a new 
    #     "unknown" sentence. The "unknown" sentence should use the same terminal symbols 
    #     that the grammar "knows", and should be parseable with the productions the grammar "knows"...    
    m1 = 0    
    m2 = 1

    # Induce grammar from a subset of the treebank parsed sentences. Allocate parsers
    cfg_earley_parser, pcfg_pchart_parser = generate_grammar_and_parsers(treebank.parsed_sents()[n1:n2])

    # Parse sentences from the treebank
    for i in range(m1,m2):
        sentence = treebank.sents()[i]  # the sentence to parse
        print("Parsing:", sentence)
        
        gold_tree = treebank.parsed_sents()[i]  # the right parse tree
        
        # Parse the sentence with parsers we define; 
        # see: 
        # http://www.nltk.org/book/ch08-extras.html
        # http://www.nltk.org/book/ch08.html
        earley(cfg_earley_parser, sentence, gold_tree) # here do not return any tree... just show all of them
        tree = pchart(pcfg_pchart_parser, sentence, gold_tree) # here get the best tree
        print("\nBEST TREE WITH PROB.: %.12e" % tree.prob())
        tree.draw()  # draw the tree


if __name__ == '__main__':
    main()
