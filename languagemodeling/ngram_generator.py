import random
from collections import defaultdict


class NGramGenerator(object):
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)

        for tokens in model._count.keys():
            if len(tokens) == self._n:
                token = tokens[-1]
                prev_tokens = tokens[:-1]
                probs[prev_tokens][token] = model.cond_prob(token, prev_tokens)

        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = {}
        for token, prob in self._probs.items():
            self._sorted_probs[token] = sorted(prob.items(), key=lambda item: item[1])

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n

        sent = []
        prev_tokens = ['<s>'] * (n - 1)
        token = self.generate_token(tuple(prev_tokens))
        while token != '</s>':
            sent.append(token)
            prev_tokens = (prev_tokens + [token])[1:]
            token = self.generate_token(prev_tokens)
        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        probs = self._sorted_probs[tuple(prev_tokens)]
        random_probability = random.random()
        i = 0
        word, probability = probs[0]
        acum = probability
        while random_probability > acum:
            i += 1
            word, probability = probs[i]
            acum += probability
        return word
