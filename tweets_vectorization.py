# -*- coding: utf-8 -*-
class TweetsVectorization:
    @staticmethod
    def get_words_dict(tweets):
        words = {}

        for tweet in tweets:
            for word in tweet:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        return words

    @staticmethod
    def get_filtered_dict(words, lower=0, upper=1000000):
        filtered_words = {}
        for word, count in words.items():
            if lower <= count <= upper:
                filtered_words[word] = count

        return filtered_words

    @staticmethod
    def get_sorted_words(words):
        return [word for word, count in sorted(words.items(), key=lambda item: item[1], reverse=True)]

    @staticmethod
    def get_vectors_of_words_indexes(tweets, words):
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
    def to_same_length(tweets, length, symbol=0):
        return list(map(lambda tweet: tweet + [symbol] * (length - len(tweet)), tweets))

    @staticmethod
    def get_max_vector_len(tweets):
        return len(max(tweets, key=len))

    @staticmethod
    def get_prepared_data(tweets_preprocessor, tweets, target, preprocess_options, words_reputation_filter, train_percentage=0.8):
        t = tweets_preprocessor.preprocess(tweets, options=preprocess_options)

        words = TweetsVectorization.get_sorted_words(
            TweetsVectorization.get_filtered_dict(
                TweetsVectorization.get_words_dict(t),
                words_reputation_filter
            )
        )

        vectors = TweetsVectorization.get_vectors_of_words_indexes(t, words)
        max_vector_len = TweetsVectorization.get_max_vector_len(vectors)
        vectors = TweetsVectorization.to_same_length(vectors, max_vector_len)
        target = list(map(lambda x: x, target))
        data_division_point = int(train_percentage * len(target))

        x_train = vectors[:data_division_point]
        y_train = target[:data_division_point]

        x_val = vectors[data_division_point:]
        y_val = target[data_division_point:]

        return x_train, y_train, x_val, y_val, words, vectors, max_vector_len
