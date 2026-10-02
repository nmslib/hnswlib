import unittest

import hnswlib
import numpy as np


class BruteforceNumericRangeTestCase(unittest.TestCase):
    def _index(self, data, space='l2'):
        index = hnswlib.BFIndex(space=space, dim=data.shape[1])
        index.init_index(max_elements=len(data))
        index.add_items(data, np.arange(1, len(data) + 1))
        return index

    def test_l2_overflow_reports_error(self):
        for dim in (2, 3, 4, 16, 17, 32, 128):
            for alpha in (2.5e19, 3e19):
                data = np.zeros((2, dim), dtype=np.float32)
                data[0, 0] = alpha
                data[1, 1] = alpha
                query = np.zeros((1, dim), dtype=np.float32)
                query[0, :2] = [0.6 * alpha, 0.8 * alpha]
                reference = np.sum(
                    (data.astype(np.float64) - query.astype(np.float64)) ** 2,
                    axis=1)
                self.assertTrue(np.isfinite(data).all())
                self.assertTrue(np.isfinite(query).all())
                self.assertLess(reference[1], reference[0])
                self.assertGreater(reference.max(), np.finfo(np.float32).max)
                index = self._index(data)
                for k in (1, 2):
                    with self.subTest(dim=dim, alpha=alpha, k=k):
                        with self.assertRaisesRegex(
                                RuntimeError, 'non-finite distance'):
                            index.knn_query(query, k=k, num_threads=1)

    def test_finite_l2_results_match_float64_reference(self):
        for dim in (2, 3, 4, 16, 17, 128):
            for alpha in (1.0, 1e10, 1e19):
                with self.subTest(dim=dim, alpha=alpha):
                    data = np.zeros((2, dim), dtype=np.float32)
                    data[0, 0] = alpha
                    data[1, 1] = alpha
                    query = np.zeros((1, dim), dtype=np.float32)
                    query[0, :2] = [0.6 * alpha, 0.8 * alpha]
                    reference = np.sum(
                        (data.astype(np.float64) -
                         query.astype(np.float64)) ** 2, axis=1)
                    index = self._index(data)
                    for k in (1, 2):
                        labels, distances = index.knn_query(
                            query, k=k, num_threads=1)
                        np.testing.assert_array_equal(
                            labels[0], (np.argsort(reference) + 1)[:k])
                        np.testing.assert_allclose(
                            distances[0], reference[labels[0] - 1],
                            rtol=2e-6, atol=0)
                        self.assertEqual(distances.dtype, np.dtype('float32'))

    def test_accumulation_overflow_reports_error(self):
        for dim in (4, 16, 17):
            with self.subTest(dim=dim):
                data = np.full((1, dim), 1e19, dtype=np.float32)
                squares = data.astype(np.float64) ** 2
                self.assertLess(squares.max(), np.finfo(np.float32).max)
                self.assertGreater(squares.sum(), np.finfo(np.float32).max)
                index = self._index(data)
                with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
                    index.knn_query(np.zeros((1, dim), dtype=np.float32),
                                    k=1, num_threads=1)

    def test_extreme_finite_coordinates_report_error(self):
        largest = np.finfo(np.float32).max
        data = np.array([[largest]], dtype=np.float32)
        query = np.array([[-largest]], dtype=np.float32)
        self.assertTrue(np.isfinite(data).all())
        self.assertTrue(np.isfinite(query).all())
        with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
            self._index(data).knn_query(query, k=1, num_threads=1)

    def test_eligible_overflow_is_not_hidden_by_a_full_heap(self):
        data = np.array([[0, 0], [1e20, 0]], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
            self._index(data).knn_query(
                np.zeros((1, 2), dtype=np.float32), k=1, num_threads=1)

    def test_filter_can_exclude_an_overflowing_candidate(self):
        data = np.array([[0, 0], [1e20, 0]], dtype=np.float32)
        query = np.zeros((1, 2), dtype=np.float32)
        index = self._index(data)
        labels, distances = index.knn_query(
            query, k=1, num_threads=1, filter=lambda label: label == 1)
        np.testing.assert_array_equal(labels, [[1]])
        np.testing.assert_array_equal(distances, [[0]])
        with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
            index.knn_query(query, k=1, num_threads=1,
                            filter=lambda label: label == 2)

    def test_error_propagates_from_parallel_queries_and_index_recovers(self):
        data = np.array([[0, 0], [1e19, 0]], dtype=np.float32)
        index = self._index(data)
        queries = np.zeros((64, 2), dtype=np.float32)
        queries[37, 0] = np.finfo(np.float32).max
        for threads in (1, 4):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
                    index.knn_query(queries, k=2, num_threads=threads)
                labels, distances = index.knn_query(
                    np.zeros((8, 2), dtype=np.float32),
                    k=2, num_threads=threads)
                np.testing.assert_array_equal(
                    labels, np.tile([1, 2], (8, 1)))
                self.assertTrue(np.isfinite(distances).all())

    def test_cosine_parallel_error_and_recovery(self):
        data = np.array([[1, 0], [0, 1]], dtype=np.float32)
        index = self._index(data, space='cosine')
        queries = np.tile(data[0], (64, 1))
        queries[37, 0] = np.nan
        for threads in (1, 4):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
                    index.knn_query(queries, k=2, num_threads=threads)
                labels, distances = index.knn_query(
                    np.tile(data[0], (8, 1)), k=2, num_threads=threads)
                np.testing.assert_array_equal(labels, np.tile([1, 2], (8, 1)))
                np.testing.assert_allclose(distances, np.tile([0, 1], (8, 1)),
                                           rtol=0, atol=1e-6)

    def test_k_zero_returns_empty_arrays_without_calling_filter(self):
        def unexpected_filter(label):
            self.fail('k=0 should not evaluate the filter')
        index = self._index(np.array([[1e20, 0]], dtype=np.float32))
        queries = np.zeros((8, 2), dtype=np.float32)
        for threads in (1, 4):
            labels, distances = index.knn_query(
                queries, k=0, num_threads=threads, filter=unexpected_filter)
            self.assertEqual(labels.shape, (8, 0))
            self.assertEqual(distances.shape, (8, 0))

    def test_nonfinite_distance_results_report_error(self):
        cases = (
            ('ip', 1e20, 1e20),     # negative infinity distance
            ('ip', -1e20, 1e20),    # positive infinity distance
            ('l2', np.nan, 0.0),    # unordered distance
        )
        for space, point, query in cases:
            with self.subTest(space=space, point=point, query=query):
                index = self._index(np.array([[point]], dtype=np.float32),
                                    space=space)
                with self.assertRaisesRegex(RuntimeError, 'non-finite distance'):
                    index.knn_query(
                        np.array([[query]], dtype=np.float32),
                        k=1, num_threads=1)
