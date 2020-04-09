import unittest
from tweets import TweetsVectorization


class TestTweetsVectorization(unittest.TestCase):
    tweets = [
        ['hello', 'world', 'hello'],
        ['hello', 'disaster', 'big'],
        ['world', 'burns']
    ]

    def test_get_vocabulary_dict(self):
        result = TweetsVectorization.get_vocabulary_dict(self.tweets)

        expected = {
            'hello': 3,
            'world': 2,
            'disaster': 1,
            'big': 1,
            'burns': 1
        }

        self.assertEqual(result, expected)

    def test_get_filtered_vocabulary_dict(self):
        result = TweetsVectorization.get_filtered_vocabulary_dict(
            {
                'hello': 2,
                'world': 2,
                'disaster': 1,
                'big': 1,
                'burns': 1
            },
            2
        )

        expected = {'hello': 2, 'world': 2}

        self.assertEqual(result, expected)

    def test_get_vocabulary_list(self):
        vocabulary = {
            'hello': 3,
            'world': 1,
            'me': 2
        }
        result_1 = TweetsVectorization.get_vocabulary_list(vocabulary)

        result_2 = TweetsVectorization.get_vocabulary_list(vocabulary, True)

        result_3 = TweetsVectorization.get_vocabulary_list(vocabulary, True, False)

        expected_1 = ['hello', 'world', 'me']
        expected_2 = ['hello', 'me', 'world']
        expected_3 = ['world', 'me', 'hello']

        self.assertEqual(result_1, expected_1)
        self.assertEqual(result_2, expected_2)
        self.assertEqual(result_3, expected_3)

    def test_get_vectors_of_vocabulary_indexes(self):
        vocabulary = TweetsVectorization.get_vocabulary_list(
            TweetsVectorization.get_filtered_vocabulary_dict(
                TweetsVectorization.get_vocabulary_dict(self.tweets),
                0
            )
        )

        result = TweetsVectorization.get_vectors_of_vocabulary_indexes(self.tweets, vocabulary)

        expected = [[0, 1, 0], [0, 2, 3], [1, 4]]

        self.assertEqual(result, expected)

    def test_get_max_vector_len(self):
        vocabulary = TweetsVectorization.get_vocabulary_list(
            TweetsVectorization.get_filtered_vocabulary_dict(
                TweetsVectorization.get_vocabulary_dict(self.tweets),
                0
            )
        )

        vectors = TweetsVectorization.get_vectors_of_vocabulary_indexes(self.tweets, vocabulary)

        result = TweetsVectorization.get_max_vector_len(vectors)

        expected = 3

        self.assertEqual(result, expected)

    def test_to_same_length(self):
        vocabulary = TweetsVectorization.get_vocabulary_list(
            TweetsVectorization.get_filtered_vocabulary_dict(
                TweetsVectorization.get_vocabulary_dict(self.tweets),
                0
            )
        )
        vectors = TweetsVectorization.get_vectors_of_vocabulary_indexes(self.tweets, vocabulary)
        max_vector_len = TweetsVectorization.get_max_vector_len(vectors)

        result_1 = TweetsVectorization.to_same_length(vectors, max_vector_len)
        result_2 = TweetsVectorization.to_same_length(vectors, 1)
        result_3 = TweetsVectorization.to_same_length(vectors, 4)

        expected_1 = [[0, 1, 0], [0, 2, 3], [1, 4, 0]]
        expected_2 = [[0], [0], [1]]
        expected_3 = [[0, 1, 0, 0], [0, 2, 3, 0], [1, 4, 0, 0]]

        self.assertEqual(result_1, expected_1)
        self.assertEqual(result_2, expected_2)
        self.assertEqual(result_3, expected_3)

    def test_get_train_test_split(self):
        train_percentage = 0.8

        result = len(TweetsVectorization.get_train_test_split(self.tweets, self.tweets, train_percentage)[0])

        expected = int(train_percentage * len(self.tweets))

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
