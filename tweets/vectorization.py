# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class TweetsVectorization:
    @staticmethod
    def get_vocabulary_dict(tweets):
        vocabulary = {}

        for tweet in tweets:
            for word in tweet:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

        return vocabulary

    @staticmethod
    def get_filtered_vocabulary_dict(vocabulary, lower=0):
        filtered_vocabulary = {}

        for word, count in vocabulary.items():
            if lower <= count:
                filtered_vocabulary[word] = count

        return filtered_vocabulary

    @staticmethod
    def get_vocabulary_list(vocabulary, sort=False, reverse=True, add_start_symbol=False, symbol='!!!!***START***!!!'):
        if sort is True:
            sorted_vocabulary = sorted(vocabulary.items(), key=lambda item: item[1], reverse=reverse)
            vocabulary_list = [word for word, count in sorted_vocabulary]
        else:
            vocabulary_list = list(vocabulary.keys())

        return [symbol] + vocabulary_list if add_start_symbol else vocabulary_list

    @staticmethod
    def get_vectors_of_vocabulary_indexes(tweets, vocabulary):
        vectors_of_indexes = []

        for tweet in tweets:
            vector = []

            for word in tweet:
                try:
                    vector.append(vocabulary.index(word))
                except ValueError:
                    pass

            vectors_of_indexes.append(vector)

        return vectors_of_indexes

    @staticmethod
    def to_same_length(tweets, length, symbol=0):
        tweets_with_same_len = []
        for tweet in tweets:
            tweet_len = len(tweet)

            if tweet_len < length:
                tweets_with_same_len.append(tweet + [symbol] * (length - tweet_len))
            elif tweet_len > length:
                tweets_with_same_len.append(tweet[:length])
            else:
                tweets_with_same_len.append(tweet)

        return tweets_with_same_len

    @staticmethod
    def get_max_vector_len(tweets):
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
    def get_train_test_split(x, y, train_percentage, shuffle_data=False):
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
    def get_vocabulary(
            tweets,
            vocabulary_filter,
            sort,
            reverse,
            add_start_symbol,
            symbol=None
    ):
        return TweetsVectorization.get_vocabulary_list(
            vocabulary=TweetsVectorization.get_filtered_vocabulary_dict(
                vocabulary=TweetsVectorization.get_vocabulary_dict(tweets),
                lower=vocabulary_filter
            ),
            add_start_symbol=add_start_symbol,
            sort=sort,
            reverse=reverse,
            symbol=symbol,
        )

    @staticmethod
    def get_prepared_data_based_on_vocabulary_indexes(
            tweets,
            target,
            vocabulary
    ):
        vectors = TweetsVectorization.get_vectors_of_vocabulary_indexes(tweets, vocabulary)
        max_vector_len = TweetsVectorization.get_max_vector_len(vectors)
        x = TweetsVectorization.to_same_length(vectors, max_vector_len)
        y = list(target)

        return x, y

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
        x_train, y_train, x_val, y_val = TweetsVectorization.get_train_test_split(vectors, target, train_percentage)

        return x_train, y_train, x_val, y_val