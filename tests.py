import unittest
import string
import random
import pandas as pd
from tweets_vectorization import TweetsVectorization
from tweets_preprocessor import TweetsPreprocessor
from inits import tweet_tokenizer, porter_stemmer, stop_words, slang_abbreviations, splitters


class TestTweetsPreprocessor(unittest.TestCase):
    preprocessor = TweetsPreprocessor(tweet_tokenizer, porter_stemmer, stop_words, slang_abbreviations, splitters)
    tweets = pd.Series([
        'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all',
        'Forest fire near La Ronge Sask. Canada',
        "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected",
        '"13,000 people receive #wildfires evacuation orders in California "',
        'Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school',
        '#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires',
        '"#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas"',
        "I'm on top of the hill and I can see a fire in the woods...",
        "There's an emergency evacuation happening now in the building across the street",
        "I'm afraid that the tornado is coming to our area...",
        'Three people died from the heat wave so far',
        'Haha South Tampa is getting flooded hah- WAIT A SECOND I LIVE IN SOUTH TAMPA WHAT AM I GONNA DO WHAT AM I GONNA DO FVCK #flooding',
        "#raining #flooding #Florida #TampaBay #Tampa 18 or 19 days. I've lost count",
        "#Flood in Bago Myanmar #We arrived Bago",
        "Damage to school bus on 80 in multi car crash #BREAKING"
    ])

    def test_remove_links(self):
        tweet = ['https://www.google.com/', 'http://localhost:8080', 'www.google.com', 'http', 'www.']

        result = TweetsPreprocessor._remove_links(tweet)

        expected = []

        self.assertEqual(result, expected)

    def test_remove_hash(self):
        tweet = ['#Hello', '#World']

        result = TweetsPreprocessor._remove_hash(tweet)

        expected = ['Hello', 'World']

        self.assertEqual(result, expected)

    def test_remove_users(self):
        tweet = ['@hello', 'M@ryan']

        result = TweetsPreprocessor._remove_users(tweet)

        expected = ['M@ryan']

        self.assertEqual(result, expected)

    def test_remove_numbers(self):
        # Before removing numbers _split_words should be called
        tweet = self.preprocessor._split_words(['1', '-2', '3.4', '0', '.24'])

        result = TweetsPreprocessor._remove_numbers(tweet)

        expected = []

        self.assertEqual(result, expected)

    def test_remove_punctuations(self):
        punct = list(string.punctuation)
        tweet = punct + [f'{random.choice(punct)}word', f'word{random.choice(punct)}', '#word', '#hello']

        result = set(TweetsPreprocessor._remove_punctuations(tweet))

        expected = {'#word', '#hello'}

        self.assertEqual(result, expected)

    def test_remove_alpha(self):
        tweet = ['hello3', '#hello', '#', "'s", '*', '2pm', '3miles', '22:30'] + list(string.punctuation)

        result = TweetsPreprocessor._remove_not_alpha(tweet)

        expected = ['#hello']

        self.assertEqual(result, expected)

    def test_remove_stop_words(self):
        tweet = ["s", 'i', "i'm", 'and', 'it', 'he'] + list(stop_words)

        result = self.preprocessor._remove_stop_words(tweet)

        expected = []

        self.assertEqual(result, expected)

    def test_split_words(self):
        tweet = list(splitters) + ['hello-world', '1.2:3/4']

        result = self.preprocessor._split_words(tweet)

        expected = ['hello', 'world', '1', '2', '3', '4']

        self.assertEqual(result, expected)

    def test_to_lower_case(self):
        tweet = ['HeLlO', 'WoRLD']

        result = TweetsPreprocessor._to_lower_case(tweet)

        expected = ['hello', 'world']

        self.assertEqual(result, expected)

    def test_stem(self):
        tweet = ['spacious', 'doing', "interesting"]

        result = self.preprocessor._stem(tweet)

        expected = ['spaciou', 'do', 'interest']

        self.assertEqual(result, expected)

    def test_unslang(self):
        tweet = ['OmG', 'fYi']

        result = self.preprocessor._unslang(tweet)

        expected = ['Oh', 'My', 'God', 'For', 'Your', 'Information']

        self.assertEqual(result, expected)

    def test_join(self):
        tweet = ['hello', 'world']

        result = self.preprocessor._join(tweet)

        expected = 'hello world'

        self.assertEqual(result, expected)

    def test_tokenizer(self):
        tweet = 'This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <-- @remy: This is waaaaayyyy too much for you!!!!!!'

        result = self.preprocessor._tokenize(tweet)

        expected = [
            'This', 'is', 'a', 'coool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows',
            '<', '>', '->', '<--', ':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_1(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_numbers': False,
                'remove_not_alpha': False
            }
        ))

        expected = [
            'deed reason thi earthquak may allah forgiv us', 'forest fire near la rong sask canada',
            'resid ask shelter place notifi offic evacu shelter place order expect',
            '13,000 peopl receiv wildfir evacu order california',
            'got sent thi photo rubi alaska smoke wildfir pour school',
            'rockyfir updat california hwi 20 close direct due lake counti fire cafir wildfir',
            'flood disast heavi rain caus flash flood street manit colorado spring area',
            'top hill see fire wood',
            'emerg evacu happen build across street',
            'afraid tornado come area',
            'three peopl die heat wave far',
            'haha south tampa flood hah wait second live south tampa gonna gonna fvck flood',
            'rain flood florida tampabay tampa 18 19 day lost count',
            'flood bago myanmar arriv bago',
            'damag school bu 80 multi car crash break'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_2(self):
        result = list(self.preprocessor.preprocess(self.tweets))

        expected = [
            'deed reason thi earthquak may allah forgiv us',
            'forest fire near la rong sask canada',
            'resid ask shelter place notifi offic evacu shelter place order expect',
            'peopl receiv wildfir evacu order california',
            'got sent thi photo rubi alaska smoke wildfir pour school',
            'rockyfir updat california hwi close direct due lake counti fire cafir wildfir',
            'flood disast heavi rain caus flash flood street manit colorado spring area',
            'top hill see fire wood',
            'emerg evacu happen build across street',
            'afraid tornado come area',
            'three peopl die heat wave far',
            'haha south tampa flood hah wait second live south tampa gonna gonna fvck flood',
            'rain flood florida tampabay tampa day lost count',
            'flood bago myanmar arriv bago',
            'damag school bu multi car crash break'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_3(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_hash': False,
                'stem': False,
            }
        ))

        expected = [
            'deeds reason #earthquake may allah forgive us',
            'forest fire near la ronge sask canada',
            'residents asked shelter place notified officers evacuation shelter place orders expected',
            'people receive #wildfires evacuation orders california',
            'got sent photo ruby #alaska smoke #wildfires pours school',
            '#rockyfire update california hwy closed directions due lake county fire #cafire #wildfires',
            '#flood #disaster heavy rain causes flash flooding streets manitou colorado springs areas',
            'top hill see fire woods',
            'emergency evacuation happening building across street',
            'afraid tornado coming area',
            'three people died heat wave far',
            'haha south tampa getting flooded hah wait second live south tampa gonna gonna fvck #flooding',
            '#raining #flooding #florida #tampabay #tampa days lost count',
            '#flood bago myanmar #we arrived bago',
            'damage school bus multi car crash #breaking'
        ]

        self.assertEqual(result, expected)


class TestTweetsVectorization(unittest.TestCase):
    tweets = [
        ['hello', 'world'],
        ['hello', 'disaster', 'big'],
        ['world', 'burns']
    ]

    def test_get_words_dict(self):
        result = TweetsVectorization.get_words_dict(self.tweets)

        expected = {
            'hello': 2,
            'world': 2,
            'disaster': 1,
            'big': 1,
            'burns': 1
        }

        self.assertEqual(result, expected)

    def test_get_filtered_dict(self):
        result = TweetsVectorization.get_filtered_dict(
            TweetsVectorization.get_words_dict(self.tweets),
            2
        )

        expected = {'hello': 2, 'world': 2}

        self.assertEqual(result, expected)

    def test_get_sorted_words(self):
        result = TweetsVectorization.get_sorted_words(
            TweetsVectorization.get_filtered_dict(
                TweetsVectorization.get_words_dict(self.tweets),
                0
            ),
        )

        expected = ['hello', 'world', 'disaster', 'big', 'burns']

        self.assertEqual(result, expected)

    def test_get_vectors_of_words_indexes(self):
        words = TweetsVectorization.get_sorted_words(
            TweetsVectorization.get_filtered_dict(
                TweetsVectorization.get_words_dict(self.tweets),
                0
            )
        )

        result = TweetsVectorization.get_vectors_of_words_indexes(self.tweets, words)

        expected = [[0, 1], [0, 2, 3], [1, 4]]

        self.assertEqual(result, expected)

    def test_get_max_vector_len(self):
        words = TweetsVectorization.get_sorted_words(
            TweetsVectorization.get_filtered_dict(
                TweetsVectorization.get_words_dict(self.tweets),
                0
            )
        )
        vectors = TweetsVectorization.get_vectors_of_words_indexes(self.tweets, words)

        result = TweetsVectorization.get_max_vector_len(vectors)

        expected = 3

        self.assertEqual(result, expected)

    def test_to_same_length(self):
        words = TweetsVectorization.get_sorted_words(
            TweetsVectorization.get_filtered_dict(
                TweetsVectorization.get_words_dict(self.tweets),
                0
            )
        )
        vectors = TweetsVectorization.get_vectors_of_words_indexes(self.tweets, words)
        max_vector_len = TweetsVectorization.get_max_vector_len(vectors)

        result = TweetsVectorization.to_same_length(vectors, max_vector_len)

        expected = [[0, 1, 0], [0, 2, 3], [1, 4, 0]]

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
