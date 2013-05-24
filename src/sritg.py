# extract ITG 

"""
By Michael Cabot (6047262) and Sander Nugteren (6042023)
"""

from collections import Counter
from nltk import Tree
import argparse
import re
import itertools


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
    alignments_file = open(alignments_file_name)
    parses_file = open(parses_file_name)
    
    for l1_parse in parses_file:
        reordered_indexes = str_to_reordered_indexes(alignments_file.next())
        parse_tree = Tree(l1_parse)
        parse_forest = generate_forest(parse_tree, 
            reordered_indexes, inv_extension)
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
        chart[span] = ['%s:%s' % (chart[span], tree.node)]
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
            for right_parents in right_parent_spans:
                right_siblings = syntax_chart[(span[1], right_parents[1])]
                for rs in right_siblings:
                    if not re.search(tokens, rs):
                        for rp in syntax_chart[right_parents]:
                            new_node = rp + '/' + rs
                            syntax_chart.setdefault(span, []).append(new_node)
            for left_parents in left_parent_spans:
                left_siblings = syntax_chart[(left_parents[0], span[0])]
                for ls in left_siblings:
                    if not re.search(tokens, ls):
                        for lp in syntax_chart[left_parents]:
                            new_node = ls + '\\' + lp
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

    for word, pos_tags in lexicon.iteritems():
        lexicon_out.write('%s\t%s\n' % (word, ' '.join(pos_tags)))

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


if __name__ == '__main__':
    main()
