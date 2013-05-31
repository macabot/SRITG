# extract ITG 

"""
By Michael Cabot (6047262) and Sander Nugteren (6042023)
"""

from collections import Counter
from nltk import Tree
import argparse
import re
import itertools
import sys
import time


def extract_sitg(alignments_file_name, parses_file_name, inv_extension):
    """Extract a stochastic inversion transduction grammar (SITG)
    from the given files.
    
    Keyword arguments:
    alignments_file_name -- name of file containing alignments
        between sentences in l1_file_name and l2_file_name
    parses_file_name -- name of file containing parse trees
        of the sentences in l1_file_name
    inv_extension -- extension denoting whether a node is inverted
        
    Returns dictionary mapping binary ITG rules to their probability and 
    dictionary mapping unary rules to their probability
    Each ITG rule is represented as the tuple (lhs, rhs), where rhs is a tuple
    of nodes."""
    binary_itg, unary_itg = extract_itg(alignments_file_name, parses_file_name, 
                                        inv_extension)
    binary_sitg = freq_to_prob(binary_itg)
    unary_sitg = freq_to_prob(unary_itg)

    return binary_sitg, unary_sitg

def freq_to_prob(grammar):
    """Convert frequencies to probabilities."""
    lhs_count = count_lhs(grammar)
    prob_grammar = {}
    for rule, rule_freq in grammar.iteritems():
        prob_grammar[rule] = float(rule_freq) / lhs_count[rule[0]]

    return prob_grammar

def count_lhs(itg):
    """Count the frequency of the left-hand-side (lhs) of the
    rules in an ITG
    
    Keyword arguments:
    itg -- Counter of itg rules
    
    Returns Counter of left-hand-side nodes"""
    lhs = Counter()
    for rule, freq in itg.iteritems():
        lhs[rule[0]] += freq

    return lhs

def extract_itg(alignments_file_name, parses_file_name, inv_extension):
    """Extract a inversion transduction grammar (ITG)
    from the given files.
    
    Keyword arguments:
    alignments_file_name -- name of file containing alignments
        between sentences in l1_file_name and l2_file_name
    parses_file_name -- name of file containing parse trees
        of the sentences in l1_file_name
    inv_extension -- extension denoting whether a node is inverted
        
    Returns a Counter of binary ITG rules and unary rules. Each ITG rule is 
    represented as the tuple (lhs, rhs), where rhs is a tuple of nodes."""
    binary_itg = Counter()
    unary_itg = Counter()
    num_lines = number_of_lines(parses_file_name)
    alignments_file = open(alignments_file_name)
    parses_file = open(parses_file_name)
    
    for i, l1_parse in enumerate(parses_file):
        if i % (num_lines/100) is 0:
            sys.stdout.write('\r%d%%' % (i*100/num_lines,))
            sys.stdout.flush()

        try: # TODO remove try/catch
            reordered_indexes = str_to_reordered_indexes(alignments_file.next())
            parse_tree = Tree(l1_parse)            
            parse_forest = generate_forest(parse_tree, 
                reordered_indexes, inv_extension)
        except:
            error_log = open('error.log', 'a')
            error_log.write('%s\n' % time.asctime())
            error_log.write('line: %s\n' % i)
            error_log.write('%s\n' % l1_parse.strip())
            error_log.write('%s\n' % reordered_indexes)
            error_log.close()
            print 'Error in parse_forest/3. See error.log'
            raise

        binary_rules, unary_rules = extract_rules(parse_forest, 
                                                  parse_tree.leaves())
        for rule in binary_rules:
            binary_itg[rule] += 1

        for rule in unary_rules:
            unary_itg[rule] += 1

    alignments_file.close()
    parses_file.close()
    return binary_itg, unary_itg

def str_to_reordered_indexes(alignment_str):
    """Return the indexes of the word in the reordered sentence."""
    string_list = alignment_str.strip().split()
    return [int(pair.split('-')[1]) for pair in string_list]
    
def generate_forest(tree, reordered_indexes, inv_extension):
    """Generate a parse forest containing all valid binary trees"""
    parse_forest = {} # condenses all possible parse tree
    words = tree.leaves()
    # initialize
    syntax_chart, _, _ = initialize_syntax_chart(tree)
    for i, word in enumerate(words):
        pos_tags = get_syntax_nodes(syntax_chart, (i, i+1), i+1)
        if len(pos_tags) == 0:
            continue

        parse_forest[(i, i+1)] = {pos_tags[0]: [(word, None, i+1)]}
            
    # expand
    for span_size in xrange(2, len(words)+1): # loop over spans sizes
        for i in xrange(len(words)-span_size+1):
            j = i+span_size
            span = (i, j)
            if not is_valid_phrase(span, reordered_indexes):
                continue

            for k in xrange(i+1, j): # k splits span [i,j) in [i,k) and [k,j)
                left_children = parse_forest.get((i, k), {})
                right_children = parse_forest.get((k, j), {})
                for left, right in itertools.product(left_children, 
                        right_children):
                    for lhs in get_syntax_nodes(syntax_chart, span, k):
                        if is_inverted(reordered_indexes, (i, k), (k, j)):
                            lhs += inv_extension

                        chart_entry = parse_forest.get(span, {})
                        chart_entry.setdefault(lhs, []).append((left, right, k))
                        parse_forest[span] = chart_entry
    
    return parse_forest

def initialize_syntax_chart(tree, chart = None, index = 0):
    """Initialize a chart with the syntax-labeled nodes in the given tree."""
    if chart == None:
        chart = {}

    if not isinstance(tree, Tree):
        return chart, (index, index + 1), index+1

    min_span, max_span = None, None
    for i, child in enumerate(tree):
        chart, child_span, index = initialize_syntax_chart(child, chart, index)
        if i == 0:  # left most child
            min_span = child_span[0]
        if i == len(tree)-1:    # right most child
            max_span = child_span[1]
    
    span = (min_span, max_span)
    if span in chart:
        chart[span] = ['%s:%s' % (chart[span][0], tree.node)]
    else:
        chart[span] = [tree.node]

    return chart, span, index

def get_syntax_nodes(syntax_chart, span, k):
    """Read or create the node for the given span from the syntax chart. If the
    node is created then it is also added to the syntax chart."""
    if span in syntax_chart:
        return syntax_chart[span]
    tokens = '[+\\/]'
    left_hand_sides = syntax_chart.get((span[0], k), [])
    right_hand_sides = syntax_chart.get((k, span[1]), [])

    for lhs, rhs in itertools.product(left_hand_sides, right_hand_sides):
        if not re.search(tokens, lhs) and not re.search(tokens, rhs):
            # '+'-rules
            new_node = lhs + '+' + rhs
            syntax_chart.setdefault(span, []).append(new_node)
        else:
            #'/' and '\'-rules
            right_parent_spans = [(i, j) for (i, j) in syntax_chart 
                                  if i is span[0] and j > span[1]]
            left_parent_spans = [(i, j) for (i, j) in syntax_chart 
                                  if i < span[0] and j is span[1]]
            for right_parent_span in right_parent_spans:
                right_siblings = syntax_chart.get((span[1], 
                                                   right_parent_span[1]), [])
                for right_sibling in right_siblings:
                    if not re.search(tokens, right_sibling):
                        for right_parent in syntax_chart[right_parent_span]:
                            new_node = right_parent + '/' + right_sibling
                            syntax_chart.setdefault(span, []).append(new_node)
            for left_parent_span in left_parent_spans:
                left_siblings = syntax_chart.get((left_parent_span[0], 
                                                  span[0]), [])
                for left_sibling in left_siblings:
                    if not re.search(tokens, left_sibling):
                        for left_parent in syntax_chart[left_parent_span]:
                            new_node = left_sibling + '\\' + left_parent
                            syntax_chart.setdefault(span, []).append(new_node)

    return syntax_chart.get(span, [])    
    
def extract_rules(parse_forest, words):
    """Extract all binary and unary rules in the parse forest."""
    binary_rules = []
    # binary rules
    for span_size in reversed(xrange(2, len(words)+1)): # loop over spans sizes
        for i in xrange(len(words)-span_size+1):
            j = i+span_size
            span = (i, j)
            for lhs, rhs_list in parse_forest.get(span, {}).iteritems():
                for left, right, _ in rhs_list:
                    binary_rules.append((lhs, (left, right)))
    # unary rules
    unary_rules = []
    for i, word in enumerate(words):
        if (i, i+1) in parse_forest:
            lhs, _ = parse_forest[(i, i+1)].items()[0]
            unary_rules.append((lhs, (word,)))

    return binary_rules, unary_rules

def is_valid_phrase(span, reordered_indexes):
    """Check whether a span is contigious"""
    index_slice = reordered_indexes[span[0]:span[1]]
    return len(index_slice)-1 == max(index_slice) - min(index_slice)

def is_inverted(reordered_indexes, span1, span2):
    """Checks if two spans are inverted according to an alignment
    
    Keyword arguments:
    alignment -- dictionary mapping index of words from line1
        to index of corresponding word in line2
    span1 -- range of left constituent
    span2 -- range of right constituent
    
    Returns True if the spans are inverted according to the alignment"""
    return reordered_indexes[span1[1]-1] > reordered_indexes[span2[0]]

def grammar_to_bitpar_files(prefix, grammar, lexicon):
    """Creates a grammar and lexicon file from an (s)itg. 
    Grammar file is formatted: <frequency> <lhs> <rhs> 
    Lexicon file is formatted: <word> <lhs1> <freq1> <lhs2> <freq2> ...
    
    Keyword arguments:
    prefix -- prefix of output file names. Grammar file and lexicon file will
        get the extension .grammar and .lexicon respectively.
    grammar -- dictionary mapping binary itg-rules to their frequency or
    probability
    lexicon -- dictionary mapping unary rules to their frequency or 
    probability."""
    grammar_out = open('%s.grammar'%prefix, 'w')
    lexicon_out = open('%s.lexicon'%prefix, 'w')
    for rule, value in grammar.iteritems():
        grammar_out.write('%s %s %s\n' % (value, rule[0], ' '.join(rule[1])))

    for rule, value in lexicon.iteritems():
        lexicon_out.write('%s\t%s\t%s\n' % (rule[1][0], rule[0], value))

    grammar_out.close()
    lexicon_out.close()

def tree_to_reordered_sentence(tree, inv_extension):
    """Reorders a sentences according to its itg-tree
    
    Keyword arguments:
    tree -- nltk tree
    
    Returns reordered string"""
    pattern = '%s' % inv_extension # match if contains string
    if not isinstance(tree, Tree): # if terminal node
        return tree
    elif len(tree)==1: # if unary rule
        return tree_to_reordered_sentence(tree[0], inv_extension)
    else:
        left_string = tree_to_reordered_sentence(tree[0], inv_extension)
        right_string = tree_to_reordered_sentence(tree[1], inv_extension)
        if re.search(pattern, tree.node): # if inverted rule
            return '%s %s' % (right_string, left_string)
        else:
            return '%s %s' % (left_string, right_string)

def reorder(reordering_file_name, output_file_name, inv_extension):
    """Reorder all sentences according to their itg parses
    
    Keyword arguments:
    reordering_file_name -- File containing ITG parses
    output_file_name -- output file for reordered sentences
    inv_extension -- extension of a node denoting the lhs of an inverted rule"""
    reordering_file = open(reordering_file_name, 'r')
    out = open(output_file_name, 'w')
    for line in reordering_file:
        line = line.strip()
        tree = Tree(line)
        reordered_sentence = tree_to_reordered_sentence(tree, inv_extension)
        out.write('%s\n' % reordered_sentence)

    reordering_file.close()
    out.close()

def main():
    """Read command line arguments and perform corresponding action"""
    arg_parser = argparse.ArgumentParser(description='Create an (S)ITG or \
        reorder sentences.')
    arg_parser.add_argument("-a", "--alignments",
        help="File containing alignments.")
    arg_parser.add_argument("-p", "--parses",
        help="File containing sentence parses.")
    arg_parser.add_argument("-s", "--stochastic", action='store_true',
        help="Calculate the probabilities of the itg rules.")
    arg_parser.add_argument("-r", "--reordering",
        help="File containing sentence parses that need to be reordered.")
    arg_parser.add_argument("-o", "--output", required=True,
        help="When constructing (S)ITG: Prefix of file names for Bitpar output.\
            When reordering: file name of reordered sentences.")
    arg_parser.add_argument("-i", "--inv_extension", default="I",
        help="Extension of a node marking it as the lhs of an inverted rule. \
        Node will be marked as <node>-<extension>")
    
    args = arg_parser.parse_args()
    # either create (S)ITG or reorder sentences
    if (bool(args.alignments) and bool(args.parses)) is bool(args.reordering):
        arg_parser.error('Invalid arguments: either construct a (S)ITG with\
            alignments (-a) and parses (-p) or reorder sentences according\
            to their itg-parses (-r).')

    output_file_name = args.output
    inv_extension = '-%s' % args.inv_extension
    if args.reordering:
        reordering_file_name = args.reordering
        reorder(reordering_file_name, output_file_name, inv_extension)
    else:
        alignments_file_name = args.alignments
        parses_file_name = args.parses
        stochastic = args.stochastic
        if stochastic:
            binary, unary = extract_sitg(alignments_file_name, parses_file_name,
                inv_extension)
        else:
            binary, unary = extract_itg(alignments_file_name, parses_file_name,
                inv_extension)

        grammar_to_bitpar_files(output_file_name, binary, unary)

def hamming_distance(translated_phrase, true_phrase):
    #computes the hamming distance between two phrases.
    #assumes that the translation has the same length as the true phrase
    translated_phrase = str.split(translated_phrase, ' ')
    true_phrase = str.split(true_phrase, ' ')
    total = 0.0
    #maybe could be neater with itertools?
    for i in xrange(len(translated_phrase)):
        if translated_phrase[i] is not true_phrase[i]:
            total += 1
    return 1 - total/len(translated_phrase)

def kendalls_tau(translated_phrase, true_phrase):
    #computes kendall's tau between two phrases.
    #assumes that the translation has the same length as the true phrase
    translated_phrase = str.split(translated_phrase, ' ')
    true_phrase = str.split(true_phrase, ' ')
    total = 0.0
    #maybe could be neater with itertools?
    for i in xrange(len(translated_phrase)):
        #for j in xrange(i+1, len(translated_phrase)): #could be faster?
        for j in xrange(len(translated_phrase)):
            if i < j and true_phrase.index(translated_phrase[i]) > \
                    true_phrase.index(translated_phrase[j]):
                total += 1
    Z = (len(translated_phrase) ** 2 - len(translated_phrase))/2
    return 1 - total/Z

def number_of_lines(file_name):
    """Counts the number of lines in a file
    
    Keywords arguments:
    file_name -- name of file
    
    Returns number of lines
    """
    amount = 0
    doc = open(file_name, 'r')
    for _ in doc:
        amount += 1

    doc.close()
    return amount

def test():
    """Testing goes here."""
    parse_tree = Tree("( (S (S (SBAR (IN as) (S (NP (PRP we)) (VP (VBP see) (NP (PRP it))))) (, ,) (NP (EX there)) (VP (MD will) (VP (VB be) (NP (NP (DT a) (NN dovetailing)) (PP (IN of) (NP (NP (NP (CD three) (NNS mechanisms)) (PP (IN in) (NP (DT the) (NN future)))) (: :) (NP (NP (NP (DT a) (NN system)) (PP (IN of) (NP (JJ independent) (JJ prior) (NN approval))) (PP (IN by) (NP (DT the) (JJ financial) (NN controller)))) (, ,) (NP (NN concomitant)) (CC and) (NP (NP (JJ follow-up) (NN control)) (PP (IN by) (NP (NP (DT the) (JJ internal) (NN audit) (NN service)) (PRN (: -) (VP (ADVP (RB also)) (VBN known) (PP (IN as) (NP (DT the) (NN audit) (NN service)))) (: -)) (SBAR (WHNP (WDT which)) (S (VP (VBZ has) (ADVP (RB yet)) (S (VP (TO to) (VP (VB be) (VP (VBN set) (PRT (RP up))))))))))))))))))) (, ,) (CC and) (S (ADVP (RB finally)) (, ,) (NP (EX there)) (VP (MD will) (VP (VB be) (NP (NP (DT the) (JJ targeted) (NN tracking-down)) (PP (IN of) (NP (NP (NNS irregularities)) (PP (IN by) (NP (NP (NNP OLAF)) (, ,) (NP (DT the) (JJ new) (JJ anti-fraud) (NN office)))))))))) (. .)))")
    reordered_indexes = [0, 1, 2, 3, 4, 15, 5, 6, 7, 10, 11, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 21, 36, 37, 43, 44, 45, 46, 47, 48, 38, 30, 31, 32, 33, 49, 50, 51, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 65, 66, 67, 68, 71]
    parse_forest = generate_forest(parse_tree, reordered_indexes, '-I')
    for k, v in parse_forest.iteritems():
        print k, v

if __name__ == '__main__':
    main()
    #test()
