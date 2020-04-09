import unittest
import string
import pandas as pd
from tweets import TweetsPreprocessor, tweet_tokenizer, porter_stemmer, stop_words, slang_abbreviations, splitters


class TestTweetsPreprocessor(unittest.TestCase):
    preprocessor = TweetsPreprocessor(
        tokenizer=tweet_tokenizer,
        stemmer=porter_stemmer,
        stop_words=stop_words,
        slang_abbreviations=slang_abbreviations,
        splitters=splitters
    )
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
        tweet = punct + ['1300', '...', '!!!', '', '!??!!*^&', '#word', '#hello']

        result = set(TweetsPreprocessor._remove_punctuations(tweet))

        expected = {'#word', '#hello', '1300'}

        self.assertEqual(result, expected)

    def test_remove_alpha(self):
        tweet = ['hello3', '#hello', '#', "'s", '*', '2pm', '3miles', '22:30', '13'] + list(string.punctuation)

        result = TweetsPreprocessor._remove_not_alpha(tweet)

        expected = ['#hello', '13']

        self.assertEqual(result, expected)

    def test_remove_stop_words(self):
        tweet = ["s", 'i', "i'm", 'and', 'it', 'he'] + list(stop_words)

        result = self.preprocessor._remove_stop_words(tweet)

        expected = []

        self.assertEqual(result, expected)

    def test_split_words(self):
        tweet = list(splitters) + ['hello-world', '1 2:3/4']

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

        expected = ['oh', 'my', 'god', 'for', 'your', 'information']

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

    def test_is_punctuation(self):
        puncts = ['##!*', '.', '&*?>!', '']

        result = [TweetsPreprocessor._is_punctuation(p) for p in puncts]

        expected = [True, True, True, True]

        self.assertEqual(result, expected)

    def test_is_empty_str(self):
        epty_strs = ['   ', ' ', '']

        result = [TweetsPreprocessor._is_empty_str(s) for s in epty_strs]

        expected = [True, True, True]

        self.assertEqual(result, expected)

    def test_is_hash_tag(self):
        hashtags = ['#', '#h', 'h#']

        result = [TweetsPreprocessor._is_hashtag(h) for h in hashtags]

        expected = [False, True, False]

        self.assertEqual(result, expected)

    def test_is_number(self):
        numbers = ['1,2', '1 000', '1.0', '01']

        result = [TweetsPreprocessor._is_number(n) for n in numbers]

        expected = [True, True, True, True]

        self.assertEqual(result, expected)

    def test_preprocess_1(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_numbers': False,
                'remove_not_alpha': False,
                'correct_spellings': True
            }
        ))

        expected = [
            'deed reason thi earthquak may allah forgiv us',
            'forest fire near la rong sask canada',
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

    def test_preprocess_4(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_hash': False,
                'stem': False,
                'remove_numbers': False,
            }
        ))

        expected = [
            'deeds reason #earthquake may allah forgive us',
            'forest fire near la ronge sask canada',
            'residents asked shelter place notified officers evacuation shelter place orders expected',
            '13,000 people receive #wildfires evacuation orders california',
            'got sent photo ruby #alaska smoke #wildfires pours school',
            '#rockyfire update california hwy 20 closed directions due lake county fire #cafire #wildfires',
            '#flood #disaster heavy rain causes flash flooding streets manitou colorado springs areas',
            'top hill see fire woods',
            'emergency evacuation happening building across street',
            'afraid tornado coming area',
            'three people died heat wave far',
            'haha south tampa getting flooded hah wait second live south tampa gonna gonna fvck #flooding',
            '#raining #flooding #florida #tampabay #tampa 18 19 days lost count',
            '#flood bago myanmar #we arrived bago',
            'damage school bus 80 multi car crash #breaking'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_5(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_hash': False,
                'stem': False,
                'remove_numbers': False,
                'remove_stop_words': False,
            }
        ))

        expected = [
            'our deeds are the reason of this #earthquake may allah forgive us all',
            'forest fire near la ronge sask canada',
            'all residents asked to shelter in place are being notified by officers no other evacuation or shelter in place orders are expected',
            '13,000 people receive #wildfires evacuation orders in california',
            'just got sent this photo from ruby #alaska as smoke from #wildfires pours into a school',
            '#rockyfire update california hwy 20 closed in both directions due to lake county fire #cafire #wildfires',
            '#flood #disaster heavy rain causes flash flooding of streets in manitou colorado springs areas',
            'i m on top of the hill and i can see a fire in the woods',
            'there s an emergency evacuation happening now in the building across the street',
            'i m afraid that the tornado is coming to our area',
            'three people died from the heat wave so far',
            'haha south tampa is getting flooded hah wait a second i live in south tampa what am i gonna do what am i gonna do fvck #flooding',
            '#raining #flooding #florida #tampabay #tampa 18 or 19 days i ve lost count',
            '#flood in bago myanmar #we arrived bago',
            'damage to school bus on 80 in multi car crash #breaking'
        ]

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()