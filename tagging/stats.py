""" Part-of-speech statistics class."""
import nltk

class Stats:
    """Several statistics for a POS tagged corpus.
    """

    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        self._tagged_sents = list(tagged_sents)
        self._words = nltk.FreqDist(
            tok for sent in self._tagged_sents for (tok, _) in sent
        )
        self._tags = nltk.FreqDist(
            tag for sent in self._tagged_sents for (_, tag) in sent
        )

        self._words_to_tags = nltk.ConditionalFreqDist(
        (word, tag) for sent in self._tagged_sents for (word, tag) in sent
        )

        self._tag_to_words = nltk.ConditionalFreqDist(
            (tag, word) for sent in self._tagged_sents for (word, tag) in sent
        )

    def sent_count(self):
        """Total number of sentences."""
        return len(self._tagged_sents)

    def token_count(self):
        """Total number of tokens."""
        return self._words.N()

    def words(self):
        """Vocabulary (set of word types)."""
        return self._words.keys()

    def word_count(self):
        """Vocabulary size."""
        return len(self.words())

    def word_freq(self, w):
        """Frequency of word w."""
        return self._words[w]

    def tags(self):
        """POS Tagset."""
        return self._tags.keys()

    def tag_count(self):
        """POS tagset size."""
        return len(self.tags())

    def tag_freq(self, t):
        """Frequency of tag t."""
        return self._tags[t]

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""

        return [
            w for w in self._words_to_tags
            if len(self._words_to_tags[w]) == 1
        ]

    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.

        n -- number of tags.
        """
        return [
            w for w in self._words_to_tags
            if len(self._words_to_tags[w]) > 1
        ]



    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self._tag_to_words[t])
