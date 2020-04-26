import unittest
import string
import pandas as pd
import numpy as np
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
        '@maryan: http://ikn.com www.goog.com Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all',
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
    locations = pd.Series([
        np.nan,
        'Canada',
        np.nan,
        'California',
        'Alaska',
        'California',
        'Colorado',
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        'South Tampa',
        'Florida',
        'Bago Myanmar',
        np.nan
    ])
    keywords = pd.Series([
        'earthquake',
        'fire',
        'evacuation',
        'wildfires',
        'wildfires',
        'fire',
        'flood',
        'fire',
        'emergency',
        'tornado',
        'died',
        'flooding',
        'flooding',
        np.nan,
        np.nan
    ])

    def test_remove_links(self):
        tweet = ['https://www.google.com/', 'http://localhost:8080', 'www.google.com', 'http', 'www.', '<link>']

        result = TweetsPreprocessor._remove_links(tweet)

        expected = ['<link>']

        self.assertEqual(result, expected)

    def test_remove_hash(self):
        tweet = ['#Hello', '#World', '<hashtag>']

        result = TweetsPreprocessor._remove_hash(tweet)

        expected = ['Hello', 'World', '<hashtag>']

        self.assertEqual(result, expected)

    def test_remove_users(self):
        tweet = ['@hello', 'M@ryan', '<user>']

        result = TweetsPreprocessor._remove_users(tweet)

        expected = ['M@ryan', '<user>']

        self.assertEqual(result, expected)

    def test_remove_numbers(self):
        # Before removing numbers _split_words should be called
        tweet = self.preprocessor._split_words(['1', '-2', '3.4', '0', '.24', '<number>'])

        result = TweetsPreprocessor._remove_numbers(tweet)

        expected = ['<number>']

        self.assertEqual(result, expected)

    def test_remove_punctuations(self):
        punct = list(string.punctuation)
        tweet = punct + ['1300', '...', '!!!', '', '!??!!*^&', '#word', '#hello', '<user>', '<url>', '<hashtag>']

        result = set(TweetsPreprocessor._remove_punctuations(tweet))

        expected = {'#word', '#hello', '1300', '<user>', '<url>', '<hashtag>'}

        self.assertEqual(result, expected)

    def test_remove_alpha(self):
        tweet = ['hello3', '#hello', '#', "'s", '*', '2pm', '3miles', '22:30', '13', '<user>', '<url>',
                 '<hashtag>', '<number>'] + list(string.punctuation)

        result = TweetsPreprocessor._remove_not_alpha(tweet)

        expected = ['#hello', '13', '<user>', '<url>', '<hashtag>', '<number>']

        self.assertEqual(result, expected)

    def test_remove_stop_words(self):
        tweet = ["s", 'i', "i'm", 'and', 'it', 'he', '<user>'] + list(stop_words)

        result = self.preprocessor._remove_stop_words(tweet)

        expected = ['<user>']

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
        tweet = ['spacious', 'doing', "interesting", '<user>']

        result = self.preprocessor._stem(tweet)

        expected = ['spaciou', 'do', 'interest', '<user>']

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

        expected = ['This', 'is', 'a', 'coool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<',
                    '>', '->', '<--', '@remy', ':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']

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
            },
            keywords=self.keywords,
            locations=self.locations
        ))

        expected = [
            'deed reason thi earthquak may allah forgiv us <url> <user> <hashtag> <keyword>',
            'forest fire near la rong sask canada <keyword> <location>',
            'resid ask shelter place notifi offic evacu shelter place order expect <keyword>',
            '13,000 peopl receiv wildfir evacu order california <hashtag> <number> <keyword> <location>',
            'got sent thi photo rubi alaska smoke wildfir pour school <hashtag> <keyword> <location>',
            'rockyfir updat california hwi 20 close direct due lake counti fire cafir wildfir <hashtag> <number> <keyword> <location>',
            'flood disast heavi rain caus flash flood street manit colorado spring area <hashtag> <keyword> <location>',
            'top hill see fire wood <keyword>',
            'emerg evacu happen build across street <keyword>',
            'afraid tornado come area <keyword>',
            'three peopl die heat wave far <keyword>',
            'haha south tampa flood hah wait second live south tampa gonna gonna fvck flood <hashtag> <keyword> <location>',
            'rain flood florida tampabay tampa 18 19 day lost count <hashtag> <number> <keyword> <location>',
            'flood bago myanmar arriv bago <hashtag> <location>',
            'damag school bu 80 multi car crash break <hashtag> <number>'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_2(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            keywords=self.keywords,
            locations=self.locations
        ))

        expected = [
            'deed reason thi earthquak may allah forgiv us <url> <user> <hashtag> <keyword>',
            'forest fire near la rong sask canada <keyword> <location>',
            'resid ask shelter place notifi offic evacu shelter place order expect <keyword>',
            'peopl receiv wildfir evacu order california <hashtag> <number> <keyword> <location>',
            'got sent thi photo rubi alaska smoke wildfir pour school <hashtag> <keyword> <location>',
            'rockyfir updat california hwi close direct due lake counti fire cafir wildfir <hashtag> <number> <keyword> <location>',
            'flood disast heavi rain caus flash flood street manit colorado spring area <hashtag> <keyword> <location>',
            'top hill see fire wood <keyword>',
            'emerg evacu happen build across street <keyword>',
            'afraid tornado come area <keyword>',
            'three peopl die heat wave far <keyword>',
            'haha south tampa flood hah wait second live south tampa gonna gonna fvck flood <hashtag> <keyword> <location>',
            'rain flood florida tampabay tampa day lost count <hashtag> <number> <keyword> <location>',
            'flood bago myanmar arriv bago <hashtag> <location>',
            'damag school bu multi car crash break <hashtag> <number>'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_3(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_links': False,
                'remove_users': False,
                'remove_not_alpha': False,
                'remove_hash': False,
                'stem': False,
            },
            keywords=self.keywords,
            locations=self.locations
        ))

        expected = [
            '@maryan ikn.com www.goog.com deeds reason #earthquake may allah forgive us <url> <user> <hashtag> <keyword>',
            'forest fire near la ronge sask canada <keyword> <location>',
            'residents asked shelter place notified officers evacuation shelter place orders expected <keyword>',
            'people receive #wildfires evacuation orders california <hashtag> <number> <keyword> <location>',
            'got sent photo ruby #alaska smoke #wildfires pours school <hashtag> <keyword> <location>',
            '#rockyfire update california hwy closed directions due lake county fire #cafire #wildfires <hashtag> <number> <keyword> <location>',
            '#flood #disaster heavy rain causes flash flooding streets manitou colorado springs areas <hashtag> <keyword> <location>',
            'top hill see fire woods <keyword>',
            'emergency evacuation happening building across street <keyword>',
            'afraid tornado coming area <keyword>',
            'three people died heat wave far <keyword>',
            'haha south tampa getting flooded hah wait second live south tampa gonna gonna fvck #flooding <hashtag> <keyword> <location>',
            '#raining #flooding #florida #tampabay #tampa days lost count <hashtag> <number> <keyword> <location>',
            '#flood bago myanmar #we arrived bago <hashtag> <location>',
            'damage school bus multi car crash #breaking <hashtag> <number>'
        ]

        self.assertEqual(result, expected)

    def test_preprocess_4(self):
        result = list(self.preprocessor.preprocess(
            self.tweets,
            options={
                'remove_hash': False,
                'stem': False,
                'remove_numbers': False,
            },
            keywords=self.keywords,
            locations=self.locations
        ))

        expected = [
            'deeds reason #earthquake may allah forgive us <url> <user> <hashtag> <keyword>',
            'forest fire near la ronge sask canada <keyword> <location>',
            'residents asked shelter place notified officers evacuation shelter place orders expected <keyword>',
            '13,000 people receive #wildfires evacuation orders california <hashtag> <number> <keyword> <location>',
            'got sent photo ruby #alaska smoke #wildfires pours school <hashtag> <keyword> <location>',
            '#rockyfire update california hwy 20 closed directions due lake county fire #cafire #wildfires <hashtag> <number> <keyword> <location>',
            '#flood #disaster heavy rain causes flash flooding streets manitou colorado springs areas <hashtag> <keyword> <location>',
            'top hill see fire woods <keyword>',
            'emergency evacuation happening building across street <keyword>',
            'afraid tornado coming area <keyword>',
            'three people died heat wave far <keyword>',
            'haha south tampa getting flooded hah wait second live south tampa gonna gonna fvck #flooding <hashtag> <keyword> <location>',
            '#raining #flooding #florida #tampabay #tampa 18 19 days lost count <hashtag> <number> <keyword> <location>',
            '#flood bago myanmar #we arrived bago <hashtag> <location>',
            'damage school bus 80 multi car crash #breaking <hashtag> <number>'
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
            },
            keywords=self.keywords,
            locations=self.locations
        ))

        expected = [
            'our deeds are the reason of this #earthquake may allah forgive us all <url> <user> <hashtag> <keyword>',
            'forest fire near la ronge sask canada <keyword> <location>',
            'all residents asked to shelter in place are being notified by officers no other evacuation or shelter in place orders are expected <keyword>',
            '13,000 people receive #wildfires evacuation orders in california <hashtag> <number> <keyword> <location>',
            'just got sent this photo from ruby #alaska as smoke from #wildfires pours into a school <hashtag> <keyword> <location>',
            '#rockyfire update california hwy 20 closed in both directions due to lake county fire #cafire #wildfires <hashtag> <number> <keyword> <location>',
            '#flood #disaster heavy rain causes flash flooding of streets in manitou colorado springs areas <hashtag> <keyword> <location>',
            'i m on top of the hill and i can see a fire in the woods <keyword>',
            'there s an emergency evacuation happening now in the building across the street <keyword>',
            'i m afraid that the tornado is coming to our area <keyword>',
            'three people died from the heat wave so far <keyword>',
            'haha south tampa is getting flooded hah wait a second i live in south tampa what am i gonna do what am i gonna do fvck #flooding <hashtag> <keyword> <location>',
            '#raining #flooding #florida #tampabay #tampa 18 or 19 days i ve lost count <hashtag> <number> <keyword> <location>',
            '#flood in bago myanmar #we arrived bago <hashtag> <location>',
            'damage to school bus on 80 in multi car crash #breaking <hashtag> <number>'
        ]

        self.assertEqual(result, expected)

    def test_add_link_flag(self):
        tweet = 'http://ing.com, www.google.com'

        result = TweetsPreprocessor._add_link_flag(tweet)

        expected = 'http://ing.com, www.google.com <url>'

        self.assertEqual(result, expected)

    def test_add_user_flag(self):
        tweet = '@maryan'

        result = TweetsPreprocessor._add_user_flag(tweet)

        expected = '@maryan <user>'

        self.assertEqual(result, expected)

    def test_add_hash_flag(self):
        tweet = '#'

        result = TweetsPreprocessor._add_hash_flag(tweet)

        expected = '# <hashtag>'

        self.assertEqual(result, expected)

    def test_add_number_flag(self):
        tweet = '2'

        result = TweetsPreprocessor._add_number_flag(tweet)

        expected = '2 <number>'

        self.assertEqual(result, expected)

    def test_is_flag(self):
        tweet = ['<url>', '<user>', '<hashtag>', '<number>', '<keyword>', '<location>']

        result = [TweetsPreprocessor._is_flag(w) for w in tweet]

        expected = [True, True, True, True, True, True]

        self.assertEqual(result, expected)

    def test_add_keyword_flags(self):
        tweets = pd.Series(['', ''])
        keywords = pd.Series(['fire', np.nan])

        result = TweetsPreprocessor._add_keyword_flags(tweets, keywords)

        expected = pd.Series([' <keyword>', ''])

        self.assertEqual(list(result), list(expected))

    def test_add_location_flags(self):
        tweets = pd.Series(['', ''])
        locations = pd.Series(['USA', np.nan])

        result = TweetsPreprocessor._add_location_flags(tweets, locations)

        expected = pd.Series([' <location>', ''])

        self.assertEqual(list(result), list(expected))


if __name__ == '__main__':
    unittest.main()
