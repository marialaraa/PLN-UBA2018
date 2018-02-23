"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] [-c <clf>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
                  memm: Maximum Entropy Markov Model
  -n <n>        Order of the model (if needed).
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import pickle

from docopt import docopt

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger

# from tagging.memm import MEMM


models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    # 'memm': MEMM,
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('../corpus/ancora/', files)
    sents = corpus.tagged_sents()

    # train the model
    model_class = models[opts['-m']]
    model = model_class(sents)

    # USEFUL FOR MODELS WITH PARAMETERS:
    # if opts['-n']:
    #     n = int(opts['-n'])
    #     if opts['-m'] == 'memm':
    #         clf = opts['-c']
    #         model = model_class(n, sents, clf=clf)
    #     else:
    #         model = model_class(n, sents)
    # else:
    #     # only for baselines
    #     model = model_class(sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
