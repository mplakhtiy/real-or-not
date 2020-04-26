import matplotlib.pyplot as plt
from data import train_data


# DATA ANALYSIS SECTION #
def draw_class_distribution_graph(data):
    plt.rcParams['figure.figsize'] = (7, 5)
    tweets_class_distribution = data.target.value_counts()
    disaster_tweets_count = tweets_class_distribution[1]
    not_disaster_tweets_count = tweets_class_distribution[0]

    plt.bar(10, disaster_tweets_count, 3, label="Disaster Tweets", color='green')
    plt.bar(15, not_disaster_tweets_count, 3, label="Not Disaster Tweets", color='red')
    plt.legend()
    plt.ylabel('Number of Examples')
    plt.title('Tweets Class Distribution')
    plt.show()


def draw_number_of_characters_graph(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    disaster_tweet_len = data[data['target'] == 1]['text'].str.len()
    not_disaster_tweet_len = data[data['target'] == 0]['text'].str.len()

    ax1.hist(disaster_tweet_len, color='green')
    ax1.set_title('Disaster Tweets')
    ax2.hist(not_disaster_tweet_len, color='red')
    ax2.set_title('Not Disaster Tweets')
    fig.suptitle('Characters in Tweets')
    plt.show()


draw_class_distribution_graph(train_data)
draw_number_of_characters_graph(train_data)

# https://www.kaggle.com/ratan123/start-from-here-disaster-tweets-eda-basic-model
# train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
# test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))
#
# ## Number of unique words in the text ##
# train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))
# test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))
#
# ## Number of characters in the text ##
# train["num_chars"] = train["text"].apply(lambda x: len(str(x)))
# test["num_chars"] = test["text"].apply(lambda x: len(str(x)))
#
# ## Number of stopwords in the text ##
# train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
# test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
#
# ## Number of punctuations in the text ##
# train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
# test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
#
# ## Number of title case words in the text ##
# train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
# test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#
# ## Number of title case words in the text ##
# train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
# test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#
# ## Average length of the words in the text ##
# train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
