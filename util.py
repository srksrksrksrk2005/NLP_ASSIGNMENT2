# Add your import statements here

import nltk
from nltk.corpus import wordnet


# Add any utility functions here
def get_wordnet_pos(treebank_tag): ## Convert NLTK POS → WordNet POS

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN