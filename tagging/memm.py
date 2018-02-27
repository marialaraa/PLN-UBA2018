import featureforge
from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from tagging.features import (History, word_lower, word_istitle, word_isupper, word_isdigit, prev_tags,
                              NextWord, PrevWord)

CLASSIFIERS = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class MEMM:
    def __init__(self, n, tagged_sents, clf='svm'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """

        self._n = n

        # 1. build the pipeline
        features = [word_lower, prev_tags, word_istitle, word_isupper, word_isdigit,
                    PrevWord(word_lower), PrevWord(word_isdigit), PrevWord(word_istitle), PrevWord(word_isupper),
                    NextWord(word_lower), NextWord(word_isdigit), NextWord(word_istitle), NextWord(word_isupper),]
        vect = featureforge.vectorizer.Vectorizer(features)

        self._pipeline = Pipeline([('vect', vect), ('clf', CLASSIFIERS.get(clf)())])

        # 2. train it
        print('Training classifier...')
        X = list(self.sents_histories(tagged_sents))
        y = list(self.sents_tags(tagged_sents))
        self._pipeline.fit(X, y)

        # 3. build known words set
        self._vocabulary = {tok for sent in tagged_sents for (tok, tag) in sent}

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            for h in self.sent_histories(sent):
                yield h

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        prev_tags = ('<s>',) * (self._n - 1)
        sent = [w for w, _ in tagged_sent]
        for i, (w, t) in enumerate(tagged_sent):
            yield History(sent, prev_tags, i)
            prev_tags = (prev_tags + (t,))[1:]

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            for t in self.sent_tags(sent):
                yield t

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        return (t for _, t in tagged_sent)

    def tag(self, sent):
        """Tag a sentence using beam inference with beam of size 1.

        sent -- the sentence.
        """
        tags = []
        prev = ('<s>',) * (self._n - 1)
        for i, w in enumerate(sent):
            h = History(sent, prev, i)
            tag = self.tag_history(h)
            tags.append(tag)
            prev = (prev + (tag,))[1:]
        return tags

    def tag_history(self, h):
        """Tag a history.

        h -- the history.
        """
        return self._pipeline.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._vocabulary
