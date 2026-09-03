import concurrent.futures
import unittest

import hnswlib
import numpy as np


class QueryWorkspaceTest(unittest.TestCase):
    @staticmethod
    def make_index(space):
        rng = np.random.default_rng(9017)
        data = rng.standard_normal((512, 16), dtype=np.float32)
        queries = rng.standard_normal((65, 16), dtype=np.float32)
        index = hnswlib.Index(space=space, dim=16)
        index.init_index(max_elements=len(data), M=16, ef_construction=100, random_seed=77)
        index.add_items(data, np.arange(len(data)), num_threads=4)
        index.set_ef(48)
        return index, queries

    def assert_query_equal(self, expected, actual):
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_spaces_and_64_65_scheduling_boundary(self):
        for space in ("l2", "ip", "cosine"):
            with self.subTest(space=space):
                index, queries = self.make_index(space)
                for rows in (64, 65):
                    expected = index.knn_query(queries[:rows], k=10, num_threads=1)
                    actual = index.knn_query(queries[:rows], k=10, num_threads=16)
                    self.assert_query_equal(expected, actual)

    def test_deletions_filter_exception_and_reuse(self):
        index, queries = self.make_index("l2")
        for label in range(0, 512, 19):
            index.mark_deleted(label)

        allowed = lambda label: label % 3 != 0
        expected = index.knn_query(queries, k=10, num_threads=1, filter=allowed)
        actual = index.knn_query(queries, k=10, num_threads=4, filter=allowed)
        self.assert_query_equal(expected, actual)
        index.set_num_threads(0)
        self.assert_query_equal(expected, index.knn_query(queries, k=10, filter=allowed))
        index.set_num_threads(4)

        class RaisedFromFilter(RuntimeError):
            pass

        def raising_filter(label):
            raise RaisedFromFilter("intentional callback failure")

        with self.assertRaises(RaisedFromFilter):
            index.knn_query(queries[:1], k=10, num_threads=1, filter=raising_filter)

        self.assert_query_equal(expected, index.knn_query(queries, k=10, num_threads=4, filter=allowed))

    def test_repeated_and_concurrent_batches(self):
        index, queries = self.make_index("cosine")
        expected = index.knn_query(queries, k=10, num_threads=1)
        for _ in range(4):
            self.assert_query_equal(expected, index.knn_query(queries, k=10, num_threads=4))

        def run_query(_):
            return index.knn_query(queries, k=10, num_threads=4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run_query, range(8)))
        for result in results:
            self.assert_query_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
