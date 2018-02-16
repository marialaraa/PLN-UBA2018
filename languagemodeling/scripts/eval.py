"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""

import math
import pickle

from docopt import docopt
from nltk.corpus import PlaintextCorpusReader

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    root = "../corpus/Outlander"
    sentences = PlaintextCorpusReader(root, 'Outlander01.txt').sents()

    # compute the cross entropy
    log_prob = model.log_prob(sentences)
    n = sum(len(sent) + 1 for sent in sentences)  # count '</s>' event
    cross_entropy = - log_prob / n
    perplexity = math.pow(2.0, cross_entropy)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(cross_entropy))
    print('Perplexity: {}'.format(perplexity))
