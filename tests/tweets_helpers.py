import unittest
from tweets import Helpers


class TestTweetsVectorization(unittest.TestCase):
    def test_get_max_vector_len(self):
        vectors = [[0, 0, 0], [0]]

        result = Helpers.get_max_vector_len(vectors)

        expected = 3

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
