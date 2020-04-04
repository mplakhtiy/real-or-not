import matplotlib.pyplot as plt
from old.utils import get_data

train_data, test_data = get_data('./data/train.csv', './data/test.csv')


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

# draw_class_distribution_graph(train_data)
# draw_number_of_characters_graph(train_data)
