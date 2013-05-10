# extract ITG 

"""
By Michael Cabot (6047262) and Sander Nugteren (6042023)
"""

from collections import Counter
from nltk import Tree

def extract_sitg(l1_file_name, l2_file_name, 
        alignments_file_name, l1_parses_file_name):
    """Extract a stochastic inversion transduction grammar (SITG)
    from the given files.
    
    Keywords arguments:
    l1_file_name -- name of file containing sentences
    l2_file_name -- name of file containing reordered sentences
    alignments_file_name -- name of file containing alignments
        between sentences in l1_file_name and l2_file_name
    l1_parses_file_name -- name of file containing parse trees
        of the sentences in l1_file_name
        
    Returns dictionary mapping ITG rules to their probability
    Each ITG rule is represented as the 3-tuple: 
    (lhs, rhs, inverted)"""
    itg = extract_itg(l1_file_name, l2_file_name, 
            alignments_file_name, l1_parses_file_name)
    lhs_count = count_lhs(itg)
    sitg = {}
    for rule, rule_freq in itg.iteritems():
        sitg[rule] = float(rule_freq) / lhs_count[rule[0]]

    return sitg

def count_lhs(itg):
    """Count the frequency of the left-hand-side (lhs) of the
    rules in an ITG
    
    Keywords arguments:
    itg -- Counter of itg rules
    
    Returns Counter of left-hand-side nodes"""
    lhs = Counter()
    for rule, freq in itg.iteritems():
        lhs[rule[0]] += freq

    return lhs

def extract_itg(l1_file_name, l2_file_name, 
        alignments_file_name, l1_parses_file_name):
    """Extract a inversion transduction grammar (ITG)
    from the given files.
    
    Keywords arguments:
    l1_file_name -- name of file containing sentences
    l2_file_name -- name of file containing reordered sentences
    alignments_file_name -- name of file containing alignments
        between sentences in l1_file_name and l2_file_name
    l1_parses_file_name -- name of file containing parse trees
        of the sentences in l1_file_name
        
    Returns a Counter of ITG rules
    Each ITG rule is represented as the 3-tuple: 
    (lhs, rhs, inverted)"""
    itg = Counter()
    l1_file = open(l1_file_name)
    l2_file = open(l2_file_name)
    alignments_file = open(alignments_file_name)
    l1_parses_file = open(l1_parses_file_name)
    
    for line1 in l1_file:
        line2 = l2_file.next()
        alignment = str_to_alignments(alignments_file.next())
        l1_parse = l1_parses_file.next()
        rules = extract_rules(Tree(l1_parse), alignment, line1, line2)
        for rule in rules:
            itg[rule] += 1

    l1_file.close()
    l2_file.close()
    alignments_file.close()
    l1_parses_file.close()
    return itg
    
def extract_rules(tree, alignment, line1, line2, rules = None):
    """Extract ITG rules from a parse tree
    
    Keywords arguments:
    tree -- nltk.Tree object
    alignment -- dictionary mapping index of words from line1
        to index of corresponding word in line2
    line1 -- sentence
    line2 -- reordered sentence
    rules -- list of ITG rules extracted thusfar
    
    Returns list of extracted ITG rules and span of each node
    """
    if rules is None:
        rules = []

    if not isinstance(tree, Tree): # when at terminal node
        index = int(tree)
        return rules, (index, index)

    child_nodes = get_child_nodes(tree)
    (_, span0) = extract_rules(tree[0], alignment, line1, line2, rules)
    inverted = False
    if len(tree) > 1:
        (_, span1) = extract_rules(tree[1], alignment, line1, line2, rules)
        inverted = is_inverted(alignment, span0, span1)

    rule = (tree.node, child_nodes, inverted)
    rules.append(rule)
    
    if len(tree) > 1:
        return rules, (span0[0], span1[1])
    else:
        return rules, span0

def get_child_nodes(tree):
    child_nodes = []
    for child in tree:
        if isinstance(child, Tree):
            child_nodes.append(child.node)
        else:
            child_nodes.append(child)
    
    return tuple(child_nodes)

def is_inverted(alignment, span1, span2):
    """Checks if two spans are inverted according to an alignment
    
    Keywords arguments:
    alignment -- dictionary mapping index of words from line1
        to index of corresponding word in line2
    span1 -- range of left constituent
    span2 -- range of right constituent
    
    Returns True if the spans are inverted according to the alignment"""
    return alignment[span1[1]] > alignment[span2[0]]

def str_to_alignments(string):
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

    
if __name__ == '__main__':
    tree = Tree('(S (NP (N 0)) (VP (V 1) (NP (N 2))))')
    alignment = {0:2, 1:1, 2:0}
    line1 = 'man bites dog'
    line2 = 'c b a'
    print extract_rules(tree, alignment, line1, line2)

