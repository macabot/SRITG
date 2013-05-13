# extract ITG 

"""
By Michael Cabot (6047262) and Sander Nugteren (6042023)
"""

from collections import Counter
from nltk import Tree
import argparse
import re

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
        alignment = str_to_alignment(alignments_file.next())
        rules = extract_rules(Tree(l1_parse), alignment, inv_extension)[0]
        for rule in rules:
            itg[rule] += 1

    alignments_file.close()
    parses_file.close()
    return itg
    
def extract_rules(tree, alignment, inv_extension, rules = None, index = 0):
    """Extract ITG rules from a parse tree
    
    Keyword arguments:
    tree -- nltk.Tree object
    alignment -- dictionary mapping index of words in a sentence
        to index of corresponding words in reordered sentence
    rules -- list of ITG rules extracted thusfar
    index -- index of leaf to encounter next
    
    Returns list of extracted ITG rules and span of each node
    """
    if rules is None:
        rules = []

    if not isinstance(tree, Tree): # when at terminal node
        return rules, (index, index), index+1, False, True

    child_nodes = get_child_nodes(tree)
    _, span0, index, child0_inverted, terminal = extract_rules(tree[0], 
        alignment, inv_extension, rules, index)
    if child0_inverted:
        child_nodes[0] += inv_extension

    inverted = False
    if len(tree) > 1:
        _, span1, index, child1_inverted, _ = extract_rules(tree[1], alignment, 
            inv_extension, rules, index)
        if child1_inverted:
            child_nodes[1] += inv_extension

        inverted = is_inverted(alignment, span0, span1)

    if inverted:
        tree.node += inv_extension # annotate inverted rules

    rule = (tree.node, tuple(child_nodes), inverted, terminal)
    rules.append(rule)
    
    if len(tree) > 1:
        return rules, (span0[0], span1[1]), index, inverted, False
    else:
        return rules, span0, index, inverted, False

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

def is_inverted(alignment, span1, span2):
    """Checks if two spans are inverted according to an alignment
    
    Keyword arguments:
    alignment -- dictionary mapping index of words from line1
        to index of corresponding word in line2
    span1 -- range of left constituent
    span2 -- range of right constituent
    
    Returns True if the spans are inverted according to the alignment"""
    return alignment[span1[1]] > alignment[span2[0]]

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
    pattern = '%s$' % inv_extension # match at end of string
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
    

def main():
    """Read command line arguments and perform corresponding action"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--alignments", required=True,
        help="File containing alignments")
    arg_parser.add_argument("-p", "--parses", required=True,
        help="File containing sentence parses")
    arg_parser.add_argument("-s", "--stochastic", action='store_true',
        help="Calculate the probabilities of the itg rules")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Prefix of file names for Bitpar output")
    
    args = arg_parser.parse_args()
    alignments_file_name = args.alignments
    parses_file_name = args.parses
    stochastic = args.stochastic
    prefix = args.output
    inv_extension = '-I'
    
    if stochastic:
        grammar = extract_sitg(alignments_file_name, parses_file_name,
            inv_extension)
    else:
        grammar = extract_itg(alignments_file_name, parses_file_name,
            inv_extension)

    grammar_to_bitpar_files(prefix, grammar)

if __name__ == '__main__':
    main()
