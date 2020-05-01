import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import original_train_data as d
from tweets import stop_words
from collections import defaultdict
from wordcloud import WordCloud
import string


def draw_class_distribution_graph(data):
    plt.rcParams['figure.figsize'] = (7, 5)
    tweets_class_distribution = data.target.value_counts()
    disaster_tweets_count = tweets_class_distribution[1]
    not_disaster_tweets_count = tweets_class_distribution[0]

    plt.bar(10, disaster_tweets_count, 3, label="Disaster Tweets")
    plt.bar(15, not_disaster_tweets_count, 3, label="Not Disaster Tweets")
    plt.legend()
    plt.ylabel('Number of Examples')
    plt.title('Tweets Class Distribution')
    plt.show()


def draw_number_of_characters_graph(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    disaster_tweet_len = data[data['target'] == 1]['text'].str.len()
    not_disaster_tweet_len = data[data['target'] == 0]['text'].str.len()

    ax1.hist(disaster_tweet_len)
    ax1.set_title('Disaster Tweets')
    ax2.hist(not_disaster_tweet_len, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Characters in Tweets')
    plt.show()


def draw_number_of_words_graph(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_len = data[data['target'] == 1]['text'].str.split().map(lambda x: len(x))
    ax1.hist(tweet_len)
    ax1.set_title('Disaster Tweets')
    tweet_len = data[data['target'] == 0]['text'].str.split().map(lambda x: len(x))
    ax2.hist(tweet_len, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Words in Tweets')
    plt.show()


def draw_avarage_word_len_graph(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    word = data[data['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='#1f77b4')
    ax1.set_title('Disaster Tweets')
    word = data[data['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Average Word Length in Tweets')
    plt.show()


def draw_location_distribution_graph(data):
    plt.rcParams['figure.figsize'] = (7, 5)
    filtered = data[data.location.notnull()]

    disaster_tweets_with_location_count = filtered[filtered.target == 1].location.size
    not_disaster_tweets_with_location_count = filtered[filtered.target == 0].location.size
    plt.bar(10, disaster_tweets_with_location_count, 3, label="Disaster Tweets")
    plt.bar(15, not_disaster_tweets_with_location_count, 3, label="Not Disaster Tweets")
    plt.legend()
    plt.ylabel('Number of Examples')
    plt.title('Tweets with Location Column Distribution')
    plt.show()


def draw_keyword_distribution_graph(data):
    plt.rcParams['figure.figsize'] = (7, 5)
    filtered = data[data.keyword.notnull()]

    disaster_tweets_with_keyword_count = filtered[filtered.target == 1].keyword.size
    not_disaster_tweets_with_keyword_count = filtered[filtered.target == 0].keyword.size
    plt.bar(10, disaster_tweets_with_keyword_count, 3, label="Disaster Tweets")
    plt.bar(15, not_disaster_tweets_with_keyword_count, 3, label="Not Disaster Tweets")
    plt.legend()
    plt.ylabel('Number of Examples')
    plt.title('Tweets with Keyword Column Distribution')
    plt.show()


def create_corpus(data, target):
    corpus = []

    for x in data[data['target'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)

    return corpus


def draw_top_10_stop_words(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)
    for word in disaster_corpus:
        if word in stop_words:
            dic_disaster[word] += 1
    for word in not_disaster_corpus:
        if word in stop_words:
            dic_not_disaster[word] += 1

    top_10_stopwords_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_stopwords_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)[:10]

    x1, y1 = zip(*top_10_stopwords_disaster)
    x2, y2 = zip(*top_10_stopwords_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Top 10 Stop Words in Tweets')
    plt.show()


def draw_punctuations_distribution(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)

    for item in (disaster_corpus):
        if item in string.punctuation:
            dic_disaster[item] += 1
    for item in (not_disaster_corpus):
        if item in string.punctuation:
            dic_not_disaster[item] += 1

    punctuations_distribution_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)
    punctuations_distribution_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)

    x1, y1 = zip(*punctuations_distribution_disaster)
    x2, y2 = zip(*punctuations_distribution_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Punctuation in Tweets')
    plt.show()


def draw_top_10_hashtags(data):
    disaster_corpus = create_corpus(data, 1)
    not_disaster_corpus = create_corpus(data, 0)

    dic_disaster = defaultdict(int)
    dic_not_disaster = defaultdict(int)
    for word in disaster_corpus:
        if word.startswith('#'):
            dic_disaster[word] += 1
    for word in not_disaster_corpus:
        if word.startswith('#'):
            dic_not_disaster[word] += 1

    top_10_hashtags_disaster = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_hashtags_not_disaster = sorted(dic_not_disaster.items(), key=lambda x: x[1], reverse=True)[:10]

    x1, y1 = zip(*top_10_hashtags_disaster)
    x2, y2 = zip(*top_10_hashtags_not_disaster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(x1, y1)
    ax1.set_title('Disaster Tweets')
    ax2.bar(x2, y2, color='#ff7f0e')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Top 10 Hashtags in Tweets')
    plt.show()


def draw_wordcloud(data, target):
    disaster_tweet = dict(data[data['target'] == target]['keyword'].value_counts())

    wordcloud = WordCloud(
        stopwords=stop_words, width=800, height=450, background_color="white"
    ).generate_from_frequencies(disaster_tweet)

    plt.figure(figsize=[10, 6])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


draw_class_distribution_graph(d)
draw_number_of_characters_graph(d)
draw_number_of_words_graph(d)
draw_avarage_word_len_graph(d)
draw_location_distribution_graph(d)
draw_keyword_distribution_graph(d)
draw_top_10_stop_words(d)
draw_punctuations_distribution(d)
draw_top_10_hashtags(d)
draw_wordcloud(d, 1)
draw_wordcloud(d, 0)
