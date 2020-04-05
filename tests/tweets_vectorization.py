import unittest
from tweets import TweetsVectorization


class TestTweetsVectorization(unittest.TestCase):
    tweets = [
        ['hello', 'world'],
        ['hello', 'disaster', 'big'],
        ['world', 'burns']
    ]

    def test_get_words_dict(self):
        result = TweetsVectorization._get_words_dict(self.tweets)

        expected = {
            'hello': 2,
            'world': 2,
            'disaster': 1,
            'big': 1,
            'burns': 1
        }

        self.assertEqual(result, expected)

    def test_get_filtered_dict(self):
        result = TweetsVectorization._get_filtered_dict(
            TweetsVectorization._get_words_dict(self.tweets),
            2
        )

        expected = {'hello': 2, 'world': 2}

        self.assertEqual(result, expected)

    def test_get_sorted_words(self):
        result = TweetsVectorization._get_sorted_words(
            TweetsVectorization._get_filtered_dict(
                TweetsVectorization._get_words_dict(self.tweets),
                0
            ),
        )

        expected = ['hello', 'world', 'disaster', 'big', 'burns']

        self.assertEqual(result, expected)

    def test_get_vectors_of_words_indexes(self):
        words = TweetsVectorization._get_sorted_words(
            TweetsVectorization._get_filtered_dict(
                TweetsVectorization._get_words_dict(self.tweets),
                0
            )
        )

        result = TweetsVectorization._get_vectors_of_words_indexes(self.tweets, words)

        expected = [[0, 1], [0, 2, 3], [1, 4]]

        self.assertEqual(result, expected)

    def test_get_max_vector_len(self):
        words = TweetsVectorization._get_sorted_words(
            TweetsVectorization._get_filtered_dict(
                TweetsVectorization._get_words_dict(self.tweets),
                0
            )
        )
        vectors = TweetsVectorization._get_vectors_of_words_indexes(self.tweets, words)

        result = TweetsVectorization._get_max_vector_len(vectors)

        expected = 3

        self.assertEqual(result, expected)

    def test_to_same_greater_length(self):
        words = TweetsVectorization._get_sorted_words(
            TweetsVectorization._get_filtered_dict(
                TweetsVectorization._get_words_dict(self.tweets),
                0
            )
        )
        vectors = TweetsVectorization._get_vectors_of_words_indexes(self.tweets, words)
        max_vector_len = TweetsVectorization._get_max_vector_len(vectors)

        result = TweetsVectorization._to_same_greater_length(vectors, max_vector_len)

        expected = [[0, 1, 0], [0, 2, 3], [1, 4, 0]]

        self.assertEqual(result, expected)

    def test_get_train_test_split(self):
        train_percentage = 0.8

        result = len(TweetsVectorization._get_train_test_split(self.tweets, self.tweets, train_percentage)[0])

        expected = int(train_percentage * len(self.tweets))

        self.assertEqual(result, expected)

    def test_to_same_smaller_length(self):
        result = TweetsVectorization._to_same_smaller_length([[1, 2, 3], [1, 2, 3]], 2)

        expected = [[1, 2], [1, 2]]

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
