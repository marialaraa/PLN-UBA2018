"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import os
import pickle

from docopt import docopt
from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    root = "../corpus/Outlander"
    books_filename = os.listdir(root)
    books_filename.remove('Outlander01.txt')
    sentences = PlaintextCorpusReader(root, books_filename).sents()

    # train the model
    models = {
        'addone': AddOneNGram,
        'ngram': NGram,
        'inter': InterpolatedNGram
    }
    model_class = models[opts.get('-m', 'ngram')]
    n = int(opts['-n'])
    model = model_class(n, sentences)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
