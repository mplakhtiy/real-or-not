import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


def get_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def remove_links(words):
    return [word for word in words if not word.startswith('http')]


def remove_hash(words):
    return [word[1:] if word.startswith('#') else word for word in words]


def remove_punct(words):
    return [word for word in words if word.isalpha()]


def remove_stop_words(words):
    return [word for word in words if word not in stop_words]


def to_lower_case(words):
    return [word.lower() for word in words]


def tokenize(tweet):
    return tweet_tokenizer.tokenize(tweet)


def preprocess(tweets):
    t = tweets.map(tokenize)
    t = t.map(remove_links)
    t = t.map(remove_hash)
    t = t.map(to_lower_case)
    t = t.map(remove_punct)
    t = t.map(remove_stop_words)

    return t


def get_words_dict(tweets):
    result = {}
    for tweet in tweets:
        for word in tweet:
            if word in result:
                result[word] += 1
            else:
                result[word] = 1

    return result


def get_filtered_dict(word_dict, lower=0, upper=1000000):
    result = {}
    for k, v in word_dict.items():
        if lower <= v <= upper:
            result[k] = v

    return result


def get_sorted_words(words_dict):
    return [k for k, v in sorted(words_dict.items(), key=lambda item: item[1], reverse=True)]


def get_vectors(tweets, words, words_count):
    vectors = []
    words_indexes = []

    for tweet in tweets:
        vector = []
        indexes = []
        tweet_set = list(set(tweet))

        for i, word in enumerate(words):
            if len(tweet_set) == 0:
                ending = [0] * (len(words) - i)
                vector = vector + ending
                break
            elif word in tweet_set:
                vector.append(1)
                indexes.append(i)
                tweet_set.remove(word)
            else:
                vector.append(0)

        vectors.append(vector)
        words_indexes.append(list(map(lambda x: 1 / x, indexes)) + [0] * (words_count - len(indexes)))

    return vectors, words_indexes
