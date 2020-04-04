# -*- coding: utf-8 -*-
class TweetsPreprocessor:
    def __init__(self, tokenizer, stemmer, stop_words, slang_abbreviations):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.slang_abbreviations = slang_abbreviations
        self.slang_abbreviations_keys = slang_abbreviations.keys()

    @staticmethod
    def _remove_links(words):
        return [word for word in words if not word.startswith(('http', 'www.'))]

    @staticmethod
    def _remove_users(words):
        return [word for word in words if not word.startswith('@')]

    @staticmethod
    def _remove_hash(words):
        return [word[1:] if word.startswith('#') else word for word in words]

    @staticmethod
    def _remove_punctuations(words):
        return [word for word in words if word.isalpha()]  # TODO removes everything what is not a word

    @staticmethod
    def _to_lower_case(words):
        return [word.lower() for word in words]

    @staticmethod
    def _join(words):
        return ' '.join(words)

    def _tokenize(self, tweet):
        return self.tokenizer.tokenize(tweet)

    def _remove_stop_words(self, words):
        return [word for word in words if word not in self.stop_words]

    def _stem(self, words):
        return [self.stemmer.stem(word) for word in words]

    def _unslang(self, words):
        new_words = []

        for word in words:
            key = word.upper()
            if key in self.slang_abbreviations_keys:
                new_words += self.tokenizer.tokenize(self.slang_abbreviations[key])
            else:
                new_words.append(word)

        return new_words

    def preprocess(self, tweets, options=None):
        if options is None:
            options = {}

        t = tweets.map(self._tokenize)

        if options.get('remove_links', True):
            t = t.map(TweetsPreprocessor._remove_links)
        if options.get('remove_users', True):
            t = t.map(TweetsPreprocessor._remove_users)
        if options.get('remove_hash', True):
            t = t.map(TweetsPreprocessor._remove_hash)
        if options.get('remove_stop_words', True):
            t = t.map(self._remove_stop_words)
        if options.get('remove_punctuations', True):
            t = t.map(TweetsPreprocessor._remove_punctuations)
        if options.get('unslang', True):
            t = t.map(self._unslang)
        if options.get('to_lower_case', True):
            t = t.map(TweetsPreprocessor._to_lower_case)
        if options.get('stem', True):
            t = t.map(self._stem)
        if options.get('join', True):
            t = t.map(TweetsPreprocessor._join)

        return t
