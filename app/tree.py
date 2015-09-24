from ete2 import Tree
import random

def get_example_tree():
    t1 = Tree("(A:1,(B:1,(E:1,D:1):0.5):0.5);")
    t2 = Tree('((((H,K)D,(F,I)G)B,E)A,((L,(N,Q)O)J,(P,S)M)C);', format=1)
    t3 = Tree( "(A:1,(B:1,(C:1,D:1):0.5):0.5);" )
    return random.choice([t1, t2, t3]).render("%%return")[0]