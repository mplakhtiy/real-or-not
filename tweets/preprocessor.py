# -*- coding: utf-8 -*-
import string
import pandas as pd
import numpy as np


class TweetsPreprocessor:
    def __init__(self, tokenizer, stemmer, stop_words, slang_abbreviations, splitters):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.slang_abbreviations = slang_abbreviations
        self.slang_abbreviations_keys = slang_abbreviations.keys()
        self.splitters = splitters

    @staticmethod
    def _add_link_flag(tweet):
        return tweet + ' <url>' if 'http' in tweet or 'www.' in tweet else tweet

    @staticmethod
    def _add_user_flag(tweet):
        return tweet + ' <user>' if any(word.startswith('@') for word in TweetsPreprocessor._split(tweet)) else tweet

    @staticmethod
    def _add_hash_flag(tweet):
        return tweet + ' <hashtag>' if '#' in tweet else tweet

    @staticmethod
    def _add_number_flag(tweet):
        return tweet + ' <number>' if any(char.isdigit() for char in tweet) else tweet

    @staticmethod
    def _add_keyword_flags(tweets, keywords):
        return pd.Series(
            [tweet if keywords[index] is np.nan else tweet + ' <keyword>' for index, tweet in tweets.items()]
        )

    @staticmethod
    def _add_location_flags(tweets, locations):
        return pd.Series(
            [tweet if locations[index] is np.nan else tweet + ' <location>' for index, tweet in tweets.items()]
        )

    @staticmethod
    def _is_flag(word):
        return word in {'<url>', '<user>', '<hashtag>', '<number>', '<location>', '<keyword>'}

    @staticmethod
    def _remove_links(words):
        return [word for word in words if not word.startswith(('http', 'www.'))]

    @staticmethod
    def _remove_users(words):
        return [word for word in words if not word.startswith('@')]

    @staticmethod
    def _remove_hash(words):
        return [word.replace('#', '') for word in words]

    @staticmethod
    def _to_lower_case(words):
        return [word.lower() for word in words]

    @staticmethod
    def _is_punctuation(word):
        chars = list(word)

        for char in chars:
            if char not in string.punctuation:
                return False

        return True

    @staticmethod
    def _is_empty_str(word):
        return word.replace(' ', '') == ''

    @staticmethod
    def _is_hashtag(word):
        return len(word) > 1 and word.startswith('#')

    @staticmethod
    def _is_number(word):
        return word.replace('.', '').replace(',', '').replace(' ', '').isdigit()

    @staticmethod
    def _remove_punctuations(words):
        return [
            w for w in words if
            not TweetsPreprocessor._is_empty_str(w) and
            not TweetsPreprocessor._is_punctuation(w)
        ]

    @staticmethod
    def _remove_not_alpha(words):
        return [
            word for word in words if
            TweetsPreprocessor._is_hashtag(word) or
            TweetsPreprocessor._is_flag(word) or
            TweetsPreprocessor._is_number(word) or
            word.isalpha()
        ]

    @staticmethod
    def _remove_numbers(words):
        return [
            word for word in words if not TweetsPreprocessor._is_number(word)
        ]

    @staticmethod
    def _join(words):
        return ' '.join(words)

    @staticmethod
    def _split(tweet):
        return tweet.split(' ')

    def _split_words(self, words):
        with_split_words = [w for w in words]

        for splitter in self.splitters:
            with_split_words = [w for sub in map(lambda w: w.split(splitter), with_split_words) for w in sub if
                                not TweetsPreprocessor._is_empty_str(w)]

        return with_split_words

    def _tokenize(self, tweet):
        return self.tokenizer.tokenize(tweet)

    def _remove_stop_words(self, words):
        return [
            word for word in words if
            TweetsPreprocessor._is_flag(word) or
            word not in self.stop_words
        ]

    def _stem(self, words):
        return [
            self.stemmer.stem(word) if not TweetsPreprocessor._is_flag(word) else word for word in words
        ]

    def _unslang(self, words):
        new_words = []

        for word in words:
            key = word.lower()
            if key in self.slang_abbreviations_keys:
                new_words += self.tokenizer.tokenize(self.slang_abbreviations[key])
            else:
                new_words.append(word)

        return new_words

    def preprocess(self, tweets, options=None, keywords=None, locations=None):
        if options is None:
            options = {}

        t = tweets.map(self._tokenize)

        t = t.map(self._join)

        if options.get('add_link_flag', True):
            t = t.map(TweetsPreprocessor._add_link_flag)

        if options.get('add_user_flag', True):
            t = t.map(TweetsPreprocessor._add_user_flag)

        if options.get('add_hash_flag', True):
            t = t.map(TweetsPreprocessor._add_hash_flag)

        if options.get('add_number_flag', True):
            t = t.map(TweetsPreprocessor._add_number_flag)

        if options.get('add_keyword_flag', True) and keywords is not None:
            t = TweetsPreprocessor._add_keyword_flags(t, keywords)

        if options.get('add_location_flag', True) and locations is not None:
            t = TweetsPreprocessor._add_location_flags(t, locations)

        t = t.map(self._split)

        if options.get('remove_links', True):
            t = t.map(TweetsPreprocessor._remove_links)

        if options.get('remove_users', True):
            t = t.map(TweetsPreprocessor._remove_users)

        if options.get('unslang', True):
            t = t.map(self._unslang)

        if options.get('split_words', True):
            t = t.map(self._split_words)

        if options.get('remove_hash', True):
            t = t.map(TweetsPreprocessor._remove_hash)

        if options.get('stem', True):
            t = t.map(self._stem)

        if options.get('remove_punctuations', True):
            t = t.map(TweetsPreprocessor._remove_punctuations)

        if options.get('remove_numbers', True):
            t = t.map(TweetsPreprocessor._remove_numbers)

        if options.get('to_lower_case', True):
            t = t.map(TweetsPreprocessor._to_lower_case)

        if options.get('remove_stop_words', True):
            t = t.map(self._remove_stop_words)

        if options.get('remove_not_alpha', True):
            t = t.map(TweetsPreprocessor._remove_not_alpha)

        if options.get('join', True):
            t = t.map(TweetsPreprocessor._join)

        return t
