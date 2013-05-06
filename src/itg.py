# extract ITG 

"""
By Michael Cabot (6047262) and Sander Nugteren (6042023)
"""

from collections import Counter
from nltk import Tree

def extract_sitg(l1_file_name, l2_file_name, 
        alignments_file_name, l1_parses_file_name):
    """TODO"""
    itg = extract_itg(l1_file_name, l2_file_name, 
            alignments_file_name, l1_parses_file_name)
    lhs_count = count_lhs(itg)
    sitg = {}
    for rule, rule_freq in itg.iteritems():
        sitg[rule] = float(rule_freq) / lhs_count[rule[0]]

    return sitg

def count_lhs(itg):
    """TODO"""
    lhs = Counter()
    for rule, freq in itg.iteritems():
        lhs[rule[0]] += freq

    return lhs

def extract_itg(l1_file_name, l2_file_name, 
        alignments_file_name, l1_parses_file_name):
    """TODO"""
    itg = Counter()
    l1_file = open(l1_file_name)
    l2_file = open(l2_file_name)
    alignments_file = open(alignments_file_name)
    l1_parses_file = open(l1_parses_file_name)
    
    for line1 in l1_file:
        line2 = l2_file.next()
        alignment = alignments_file.next()
        l1_parse = l1_parses_file.next()        
        tree = Tree(l1_parse)
        rules = extract_rules(tree, alignment, line1, line2)
        for rule in rules:
            itg[rule] += 1

    return itg
    
def extract_rules(tree, alignment, line1, line2, rules = None):
    """TODO"""
    if not rules:
        rules = []

    if not isinstance(tree, Tree): # when at terminal node
        return rules, (tree, tree+1)

    child_nodes = [child.node for child in tree]
    (_, span0) = extract_rules(child_nodes[0], alignment, line1, line2, rules)
    (_, span1) = extract_rules(child_nodes[1], alignment, line1, line2, rules)

    rule = (tree.node, child_nodes, inverted(span0, span1))
    rules.append(rule)
    
    return rules, (span0[0], span1[1])
    
if __name__ == '__main__':
    pass
