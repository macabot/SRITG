# extract ITG 
# TODO can a chart position contain multiple nodes?
#   1  2  3  4
# 0 a  e  a+f or e+c?
# 1    b  f
# 2       c
# 3          d
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
        
    Returns dictionary mapping ITG rules to their probability
    Each ITG rule is represented as the 3-tuple: 
    (lhs, rhs, inverted)"""
    itg = extract_itg(alignments_file_name, parses_file_name, inv_extension)
    lhs_count = count_lhs(itg)
    sitg = {}
    for rule, rule_freq in itg.iteritems():
        sitg[rule] = float(rule_freq) / lhs_count[rule[0]]

    return sitg

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
        
    Returns a Counter of ITG rules
    Each ITG rule is represented as the 3-tuple: 
    (lhs, rhs, inverted)"""
    itg = Counter()
    alignments_file = open(alignments_file_name)
    parses_file = open(parses_file_name)
    
    for l1_parse in parses_file:
        reordered_indexes = str_to_reordered_indexes(alignments_file.next())
        parse_forest = generate_forest(Tree(l1_parse), 
            reordered_indexes, inv_extension)
        rules = extract_rules(parse_forest)
        for rule in rules:
            itg[rule] += 1

    alignments_file.close()
    parses_file.close()
    return itg

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
        pos_tag = get_syntax_nodes(syntax_chart, (i, i+1))
        if pos_tag == None:
            continue

        parse_forest[(i, i+1)] = (pos_tag, [(word, None, i+1)])
            
    # expand
    for span_size in xrange(2, len(words)+1): # loop over spans sizes
        for i in xrange(len(words)-span_size+1):
            j = i+span_size
            span = (i, j)
            if not is_valid_phrase(span, reordered_indexes):
                continue

            for k in xrange(i+1, j): # k splits span [i,j) in [i,k), [k,j)
                left, _ = parse_forest.get((i, k), (None, []))
                right, _ = parse_forest.get((k, j), (None, []))
                if not left or not right:
                    continue

                if span in parse_forest:
                    lhs, rhs_list = parse_forest[span]
                    rhs_list.append((left, right, k))
                    parse_forest[span] = (lhs, rhs_list)
                else:
                    lhs = get_syntax_nodes(syntax_chart, span)
                    if lhs == None:
                        continue
                    if is_inverted(reordered_indexes, (i, k), (k, j)):
                        lhs += inv_extension

                    parse_forest[span] = (lhs, [(left, right, k)])
    
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
        chart[span] = '%s:%s' % (chart[span], tree.node)
    else:
        chart[span] = tree.node

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
            right_parent_spans = [(i,j) for (i,j) in syntax_chart if i is span[0] and j > span[1]]
            left_parent_spans = [(i,j) for (i,j) in syntax_chart if i < span[0] and j is span[1]]
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
    return syntax_chart[span]

    return 'X'

def extract_rules(parse_forest):
    """Extract all rules in the parse forest."""
    return [] # TODO implement  # extract all rules

def is_valid_phrase(span, reordered_indexes):
    """Check whether a span is contigious"""
    index_slice = reordered_indexes[span[0]:span[1]]
    return len(index_slice)-1 == max(index_slice) - min(index_slice)

def phrase_alignment_expansions(phrase_alignments, max_length = float('inf')):
    # TODO fix
    """For each language find the words that are not covered with the given
    phrase alignment.
    E.g. phrase_alignments = [(0,0), (2,0)]
    returns [1], []
    because index 1 in sentence 1 is not covered.
    
    Keyword arguments:
    phrase_alignments -- list of 2-tuples denoting the alignment between words
    max_length -- maximum length of a phrase alignment
    
    Returns 2 lists of indexes that are not covered
    """
    min1, min2, max1, max2 = phrase_range(phrase_alignments)
    if max1-min1+1 > max_length or max2-min2+1 > max_length:
        return [], []

    range1 = range(min1, max1+1)
    range2 = range(min2, max2+1)
    for a1, a2 in phrase_alignments:
        if a1 in range1:
            range1.remove(a1)
        if a2 in range2:
            range2.remove(a2)

    return range1, range2

def phrase_range(phrase_alignments):
    """Calcualte the range of a phrase alignment
    
    Keyword arguments:
    phrase_alignments -- dictionary mapping the alignment between words
    
    Returns a 4-tuples denoting the range of the phrase alignment
    """
    min1 = min2 = float('inf')
    max1 = max2 = float('-inf')
    for a1, a2 in phrase_alignments.iteritems():
        if a1 < min1:
            min1 = a1
        if a1 > max1:
            max1 = a1
        if a2 < min2:
            min2 = a2
        if a2 > max2:
            max2 = a2

    return min1, min2, max1, max2

def get_child_nodes(tree):
    """Gets the nodes of the children of the given tree
    
    Keyword arguments:
    tree -- nltk.Tree
    
    Returns nodes of the tree's children"""
    child_nodes = []
    for child in tree:
        if isinstance(child, Tree):
            child_nodes.append(child.node)
        else:
            child_nodes.append(child)
    
    return child_nodes

def is_inverted(reordered_indexes, span1, span2):
    """Checks if two spans are inverted according to an alignment
    
    Keyword arguments:
    alignment -- dictionary mapping index of words from line1
        to index of corresponding word in line2
    span1 -- range of left constituent
    span2 -- range of right constituent
    
    Returns True if the spans are inverted according to the alignment"""
    return reordered_indexes[span1[1]-1] > reordered_indexes[span2[0]]

def str_to_alignment(string):
    """Parse an alignment from a string
    
    Keyword arguments:
    string -- contains alignment
    
    Return a dictionary mapping the index of a word in language 1
        to the index of the corresponding word in language 2
    """
    string_list = string.strip().split()
    alignments = {}
    for a_str in string_list:
        a1_str, a2_str = a_str.split('-')
        alignments[int(a1_str)] = int(a2_str)

    return alignments

def remove_lexicon(grammar):
    """Removes all lexical rules from a grammar and puts them in new dictionary
    mapping word to a list of lhs and frequencies (both are strings)
    
    Keyword arguments:
    grammar -- Counter of rules: (lsh, rhs, interted, terminal)
    
    Return new grammar and lexical rules"""
    new_grammar = {}
    lexicon = {}
    for rule, freq in grammar.iteritems():
        if rule[3]: # if lexical rule
            lexicon.setdefault(rule[1][0], []).extend([rule[0], str(freq)])
        else:
            new_grammar[rule] = freq

    return new_grammar, lexicon

def grammar_to_bitpar_files(prefix, grammar):
    """Creates a grammar and lexicon file from an (s)itg. 
    Grammar file is formatted: <frequency> <lhs> <rhs> 
    Lexicon file is formatted: <word> <lhs1> <freq1> <lhs2> <freq2> ...
    
    Keyword arguments:
    prefix -- prefix of output file names. Grammar file and lexicon file will
        get the extension .grammar and .lexicon respectively.
    grammar -- dictionary mapping itg-rules to their frequency or probability"""
    grammar, lexicon = remove_lexicon(grammar)
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
            grammar = extract_sitg(alignments_file_name, parses_file_name,
                inv_extension)
        else:
            grammar = extract_itg(alignments_file_name, parses_file_name,
                inv_extension)

        grammar_to_bitpar_files(output_file_name, grammar)

def test():
    '''
    d = {}
    d[(0,1)] = ['A']
    d[(1,2)] = ['B']
    print get_syntax_nodes(d, (0,2), 1)[0]
    d[(2,3)] = ['C']
    d[(3,4)] = ['D']
    d[(0,4)] = ['S']
    print get_syntax_nodes(d, (0,3), 2)[0]
    '''
    d = {}
    d[(3,4)] = ['A']
    d[(2,3)] = ['B']
    print get_syntax_nodes(d, (2,4), 3)[0]
    d[(1,2)] = ['C']
    d[(0,1)] = ['D']
    d[(0,4)] = ['S']
    print get_syntax_nodes(d, (1,4), 2)[0]

if __name__ == '__main__':
    #main()
    test()
