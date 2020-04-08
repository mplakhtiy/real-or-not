# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class TweetsVectorization:
    @staticmethod
    def _get_words_dict(tweets):
        words = {}

        for tweet in tweets:
            for word in tweet:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        return words

    @staticmethod
    def _get_filtered_dict(words, lower=0, upper=1000000):
        filtered_words = {}
        for word, count in words.items():
            if lower <= count <= upper:
                filtered_words[word] = count

        return filtered_words

    @staticmethod
    def _get_sorted_words(words, add_start_symbol=False, symbol='!!!!***START***!!!'):
        sorted_words = [word for word, count in sorted(words.items(), key=lambda item: item[1], reverse=True)]

        return [symbol] + sorted_words if add_start_symbol else sorted_words

    @staticmethod
    def _get_vectors_of_words_indexes(tweets, words):
        words_indexes = []

        for tweet in tweets:
            indexes = []
            tweet_words = list(set(tweet))

            for i, word in enumerate(words):
                if len(tweet_words) == 0:
                    break
                elif word in tweet_words:
                    indexes.append(i)
                    tweet_words.remove(word)

            words_indexes.append(indexes)

        return words_indexes

    @staticmethod
    def to_same_greater_length(tweets, length, symbol=0):
        return list(map(lambda tweet: tweet + [symbol] * (length - len(tweet)), tweets))

    @staticmethod
    def _get_max_vector_len(tweets):
        return len(max(tweets, key=len))

    @staticmethod
    def randomize(x, y):
        perm = np.random.permutation(len(x))
        new_x = []
        new_y = []

        for index in perm:
            new_x.append(x[index])
            new_y.append(y[index])

        return new_x, new_y

    @staticmethod
    def _get_train_test_split(x, y, train_percentage, shuffle_data=False):
        if shuffle_data is False:
            data_division_point = int(train_percentage * len(y))
            x_train = x[:data_division_point]
            y_train = y[:data_division_point]

            x_val = x[data_division_point:]
            y_val = y[data_division_point:]
        else:
            true_x = []
            false_x = []

            for i, t in enumerate(x):
                if y[i] == 0:
                    false_x.append(t)
                else:
                    true_x.append(t)

            true_data_division_point = int(train_percentage * len(true_x))
            false_data_division_point = int(train_percentage * len(false_x))

            random.shuffle(true_x)
            random.shuffle(false_x)

            true_x_train = true_x[:true_data_division_point]
            true_x_val = true_x[true_data_division_point:]
            false_x_train = false_x[:false_data_division_point]
            false_x_val = false_x[false_data_division_point:]

            x_train = true_x_train + false_x_train
            y_train = [1] * len(true_x_train) + [0] * len(false_x_train)
            x_val = true_x_val + false_x_val
            y_val = [1] * len(true_x_val) + [0] * len(false_x_val)

            x_train, y_train = TweetsVectorization.randomize(x_train, y_train)
            x_val, y_val = TweetsVectorization.randomize(x_val, y_val)

        return x_train, y_train, x_val, y_val

    @staticmethod
    def to_same_smaller_length(tweets, length):
        return list(map(lambda tweet: tweet[:length], tweets))

    @staticmethod
    def get_prepared_data_based_on_words_indexes(
            tweets_preprocessor,
            tweets,
            target,
            preprocess_options,
            tweets_for_vocabulary_base=None,
            words_reputation_filter=0,
            train_percentage=0.8,
            add_start_symbol=False,
            shuffle_data=False
    ):
        t = tweets_preprocessor.preprocess(tweets, options=preprocess_options)

        if tweets_for_vocabulary_base is None:
            vocabulary_base = t
        else:
            vocabulary_base = tweets_preprocessor.preprocess(tweets_for_vocabulary_base, options=preprocess_options)

        vocabulary = TweetsVectorization._get_sorted_words(
            TweetsVectorization._get_filtered_dict(
                TweetsVectorization._get_words_dict(vocabulary_base),
                words_reputation_filter
            ),
            add_start_symbol
        )

        vectors = TweetsVectorization._get_vectors_of_words_indexes(t, vocabulary)
        max_vector_len = TweetsVectorization._get_max_vector_len(vectors)
        vectors = TweetsVectorization.to_same_greater_length(vectors, max_vector_len)
        target = list(map(lambda x: x, target))
        x_train, y_train, x_val, y_val = TweetsVectorization._get_train_test_split(
            vectors, target, train_percentage, shuffle_data=shuffle_data
        )

        return x_train, y_train, x_val, y_val, vocabulary, max_vector_len

    @staticmethod
    def get_prepared_data_based_on_count_vectorizer(
            tweets_preprocessor,
            tweets,
            target,
            preprocess_options,
            train_percentage=0.8,
            count_vectorizer_options=None
    ):
        if count_vectorizer_options is None:
            count_vectorizer_options = {
                'analyzer': 'word',
                'binary': True
            }

        t = tweets_preprocessor.preprocess(tweets, options=preprocess_options)

        vectorizer = CountVectorizer(**count_vectorizer_options)
        vectors = vectorizer.fit_transform(t).todense()

        target = target.values
        x_train, y_train, x_val, y_val = TweetsVectorization._get_train_test_split(vectors, target, train_percentage)

        return x_train, y_train, x_val, y_val
