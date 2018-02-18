# https://docs.python.org/3/library/collections.html

import itertools
import math
from collections import defaultdict
from itertools import repeat


class LanguageModel(object):
    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


class NGram(LanguageModel):
    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        sents = [self.add_separators(sent) for sent in sents]

        for sent in sents:
            for i in range(len(sent) - n + 1):
                count[tuple(sent[i:i + n])] += 1
                count[tuple(sent[i:i + n - 1])] += 1

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prev_tokens = tuple(prev_tokens) if prev_tokens else tuple()
        tokens = prev_tokens + (token,)
        return self._count.get(tokens, 0) / self._count.get(prev_tokens, 1)

    def sentence_probability(self, sent, prob_acum, get_prob):
        sent = self.add_separators(sent)
        for i in range(self._n - 1, len(sent)):
            token = sent[i]
            prev_tokens = sent[i - self._n + 1:i] if self._n > 1 else None
            prob_acum = get_prob(token, prev_tokens, prob_acum)
        return prob_acum

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        prob = lambda token, prev_tokens, acum: acum * self.cond_prob(token, prev_tokens)
        return self.sentence_probability(sent, 1, prob)

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """

        log_prob = lambda token, prev_tokens, acum: acum + math.log2(
            self.cond_prob(token, prev_tokens)) if self.cond_prob(token, prev_tokens) != 0 else -math.inf
        return self.sentence_probability(sent, 0, log_prob)

    def add_separators(self, sent):
        return list(repeat('<s>', self._n - 1)) + sent + ['</s>']


class AddOneNGram(NGram):
    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = set(list(itertools.chain.from_iterable(sents)) + ['</s>'])

        self._V = len(self._voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        prev_tokens = tuple(prev_tokens) if prev_tokens else tuple()
        tokens = prev_tokens + (token,)
        return (self._count.get(tokens, 0) + 1) / (self._count.get(prev_tokens, 1) + self._V)


class InterpolatedNGram(NGram):
    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        self._models = []
        for k in range(1, self._n + 1):
            self._models.append(NGram(k, train_sents))

        # compute vocabulary size for add-one in the last step
        if addone:
            print('Computing vocabulary...')
            self._addone = AddOneNGram(1, train_sents)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # use grid search to choose gamma
            min_gamma, min_p = None, float('inf')

            for gamma in range(1000, 10000, 100):
                self._gamma = gamma
                p = self.perplexity(held_out_sents)
                print('  {} -> {}'.format(gamma, p))

                if p < min_p:
                    min_gamma, min_p = gamma, p

            print('  Choose gamma = {}'.format(min_gamma))
            self._gamma = min_gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._models[len(tokens[:-1])]._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        else:
            prev_tokens = tuple(prev_tokens)
        assert len(prev_tokens) == n - 1

        lambdas = self.lambdas(n, tuple(prev_tokens))
        prob = 0.0
        tokens = prev_tokens + (token,)
        for i in range(len(tokens)):
            prev_token = tuple(tokens[i:-1])
            if i == n and self._addone:
                cond_ml = self._addone.cond_prob(token, prev_token)
            else:
                cond_ml = self._models[len(prev_token)].cond_prob(token, prev_token)
            prob += lambdas[i] * cond_ml
        return prob

    def lambdas(self, n, tokens):
        lambdas = []
        for i in range(n - 1):
            temp_token = tokens[i:]
            sum_olds_lambdas = 1 - sum(lambdas[:i - 1])
            count_tmp_token = self._models[len(temp_token)].count(temp_token)
            lambdas.append(sum_olds_lambdas * count_tmp_token / (count_tmp_token + self._gamma))
        lambdas.append(1 - sum(lambdas))  # last case has a special condition
        return lambdas
