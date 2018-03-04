import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}

TWEET_TOKENIZER = TweetTokenizer()
SPANISH_STEMMER = SpanishStemmer(ignore_stopwords=True)


def tweet_tokenizer(sentence):
    return TWEET_TOKENIZER.tokenize(sentence)


def stemming_tokenizer(sentence):
    return [SPANISH_STEMMER.stem(token) for token in sentence.split()]


def custom_tokenizer(sentence):
    return [SPANISH_STEMMER.stem(token) for token in TWEET_TOKENIZER.tokenize(sentence)]


def clean_tweet_text(sentences):
    mentions = r'(?:@[^\s]+\s)'  # Eliminar menciones de usuarios.
    urls = r'(?:https?\://t.co/[\w]+\s)'  # Eliminar URLs.
    duplicated_vocals = r'([a,e,i,o,u])\1+\s'  # Contraer repeticiones de 3 o más vocales.
    return [re.sub(duplicated_vocals, r'\1', re.sub(urls, '', re.sub(mentions, '', sentence))) for sentence in
            sentences]


class SentimentClassifier(object):
    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer()),
            # ('vect', CountVectorizer(tokenizer=stemming_tokenizer)),
            # ('vect', CountVectorizer(stop_words=stopwords.words('spanish'))),
            # ('vect', CountVectorizer(tokenizer=tweet_tokenizer)),
            # ('vect', CountVectorizer(tokenizer=custom_tokenizer, stop_words=stopwords.words('spanish'))),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, sentences, polarities):
        # sentences = clean_tweet_text(sentences)
        self._pipeline.fit(sentences, polarities)

    def predict(self, sentences):
        return self._pipeline.predict(sentences)
