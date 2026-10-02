import itertools
import tempfile
import unittest
import warnings

import numpy as np

import hnswlib

class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):

        data1 = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 1],
                            [1, 1, 1],
                            ])

        for space, expected_distances in [
            ('l2', [[0., 1., 2., 2., 2.]]),
            ('ip', [[-2., -1., 0., 0., 0.]]),
            ('cosine', [[0, 1.835e-01, 4.23e-01, 4.23e-01, 4.23e-01]])]:

            for rightdim in range(1, 128, 3):
                for leftdim in range(1, 32, 5):
                    data2 = np.concatenate(
                        [np.zeros([data1.shape[0], leftdim]), data1, np.zeros([data1.shape[0], rightdim])], axis=1)
                    dim = data2.shape[1]
                    p = hnswlib.Index(space=space, dim=dim)
                    p.init_index(max_elements=5, ef_construction=100, M=16)

                    p.set_ef(10)

                    p.add_items(data2)

                    # Query the elements for themselves and measure recall:
                    labels, distances = p.knn_query(np.asarray(data2[-1:]), k=5)


                    diff=np.mean(np.abs(distances-expected_distances))
                    self.assertAlmostEqual(diff, 0, delta=1e-3)


class CosineFiniteInputTestCase(unittest.TestCase):
    index_types = (hnswlib.Index, hnswlib.BFIndex)
    non_finite_values = (np.nan, np.inf, -np.inf)

    def make_index(self, index_type, num_threads=1):
        index = index_type(space='cosine', dim=4)
        index.init_index(max_elements=64)
        index.set_num_threads(num_threads)
        if index_type is hnswlib.Index:
            index.set_ef(64)
        return index

    def testNonFiniteComponentsRejected(self):
        cases = itertools.product(self.index_types, ('add_items', 'knn_query'),
                                  (1, 2), (0, 1, 3), self.non_finite_values)
        for index_type, operation, ndim, component, value in cases:
            with self.subTest(index=index_type.__name__, operation=operation,
                              ndim=ndim, component=component, value=value):
                index = self.make_index(index_type)
                if operation == 'knn_query':
                    index.add_items([[1, 0, 0, 0]])
                data = np.array([1, 0, 0, 0], dtype=np.float32)
                data[component] = value
                if ndim == 2:
                    data = data.reshape(1, 4)
                with self.assertRaises(ValueError):
                    getattr(index, operation)(data)

    def testRejectedBatchLeavesExistingItemsUnchanged(self):
        cases = itertools.product(self.index_types, (1, 4), (0, 16, 31),
                                  self.non_finite_values)
        for index_type, num_threads, invalid_row, value in cases:
            with self.subTest(index=index_type.__name__, num_threads=num_threads,
                              invalid_row=invalid_row, value=value):
                index = self.make_index(index_type, num_threads)
                index.add_items([[1, 0, 0, 0], [-1, 0, 0, 0]], [10, 11])
                data = np.tile(np.array([0, 1, 0, 0], dtype=np.float32), (32, 1))
                # A valid first row would overwrite label 10 if added prematurely.
                data[0] = [-1, 0, 0, 0]
                data[invalid_row, 3] = value
                ids = np.arange(100, 132)
                ids[0] = 10
                with self.assertRaises(ValueError):
                    index.add_items(data, ids)
                self.assertEqual(index.get_current_count(), 2)
                labels, distances = index.knn_query([[1, 0, 0, 0]], k=2)
                np.testing.assert_array_equal(labels, [[10, 11]])
                np.testing.assert_allclose(distances, [[0, 2]], atol=2e-6, rtol=0)

    def testRejectedInitialBatchPreservesAutomaticLabels(self):
        cases = itertools.product(self.index_types, (1, 4), self.non_finite_values)
        for index_type, num_threads, value in cases:
            with self.subTest(index=index_type.__name__, num_threads=num_threads,
                              value=value):
                index = self.make_index(index_type, num_threads)
                data = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (32, 1))
                data[-1, 3] = value
                with self.assertRaises(ValueError):
                    index.add_items(data)
                self.assertEqual(index.get_current_count(), 0)
                index.add_items([[1, 0, 0, 0]])
                labels, distances = index.knn_query([[1, 0, 0, 0]])
                np.testing.assert_array_equal(labels, [[0]])
                np.testing.assert_allclose(distances, [[0]], atol=2e-6, rtol=0)

    def testRejectedQueryDoesNotCallFilter(self):
        cases = itertools.product(self.index_types, (1, 4), (0, 16, 31),
                                  self.non_finite_values)
        for index_type, num_threads, invalid_row, value in cases:
            with self.subTest(index=index_type.__name__, num_threads=num_threads,
                              invalid_row=invalid_row, value=value):
                index = self.make_index(index_type, num_threads)
                index.add_items([[1, 0, 0, 0], [-1, 0, 0, 0]], [10, 11])
                queries = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (32, 1))
                queries[invalid_row, 3] = value
                visited_labels = []

                def allow_label(label):
                    visited_labels.append(label)
                    return True

                with self.assertRaises(ValueError):
                    index.knn_query(queries, num_threads=num_threads,
                                    filter=allow_label)
                self.assertEqual(visited_labels, [])
                labels, distances = index.knn_query([[1, 0, 0, 0]], k=2,
                                                   num_threads=1)
                np.testing.assert_array_equal(labels, [[10, 11]])
                np.testing.assert_allclose(distances, [[0, 2]], atol=2e-6, rtol=0)

    def testFloat64OverflowRejected(self):
        cases = itertools.product(self.index_types, ('add_items', 'knn_query'),
                                  (-1, 1))
        for index_type, operation, sign in cases:
            with self.subTest(index=index_type.__name__, operation=operation,
                              sign=sign):
                index = self.make_index(index_type)
                if operation == 'knn_query':
                    index.add_items([[1, 0, 0, 0]])
                too_large = sign * float(np.finfo(np.float32).max) * 2
                data = np.array([[1, too_large, 0, 0]], dtype=np.float64)
                self.assertTrue(np.isfinite(data).all())
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='overflow encountered in cast',
                                            category=RuntimeWarning)
                    with self.assertRaises(ValueError):
                        getattr(index, operation)(data)

    def testFiniteStridedAndZeroInputs(self):
        cases = itertools.product(self.index_types, (1, 2))
        for index_type, ndim in cases:
            with self.subTest(index=index_type.__name__, ndim=ndim):
                index = self.make_index(index_type)
                backing = np.zeros((3, 8), dtype=np.float32)
                backing[0, 0] = 1
                backing[1, 0] = -1
                data = backing[:, ::2]
                index.add_items(data, [0, 1, 2])
                queries = data[0] if ndim == 1 else data
                labels, distances = index.knn_query(queries, k=3)
                # Literal distances: same/opposite axes are 0/2; zero is 1.
                expected = np.array([[0, 2, 1], [2, 0, 1], [1, 1, 1]])
                for row in range(labels.shape[0]):
                    np.testing.assert_array_equal(np.sort(labels[row]), [0, 1, 2])
                    np.testing.assert_allclose(distances[row], expected[row, labels[row]],
                                               atol=2e-6, rtol=0)

    def testSaveLoadAfterRejectedInsertion(self):
        cases = itertools.product(self.index_types, self.non_finite_values)
        for index_type, value in cases:
            with self.subTest(index=index_type.__name__, value=value):
                index = self.make_index(index_type)
                index.add_items([[1, 0, 0, 0], [-1, 0, 0, 0]], [10, 11])
                invalid = np.array([[0, 1, value, 0]], dtype=np.float32)
                with self.assertRaises(ValueError):
                    index.add_items(invalid, [12])
                with tempfile.TemporaryDirectory() as directory:
                    filename = directory + '/cosine.bin'
                    index.save_index(filename)
                    restored = index_type(space='cosine', dim=4)
                    restored.load_index(filename)
                    restored.set_num_threads(1)
                    if index_type is hnswlib.Index:
                        restored.set_ef(64)
                    self.assertEqual(restored.get_current_count(), 2)
                    with self.assertRaises(ValueError):
                        restored.add_items(invalid, [12])
                    restored.add_items([[0, 1, 0, 0]], [12])
                    labels, distances = restored.knn_query([[1, 0, 0, 0]], k=3)
                    np.testing.assert_array_equal(labels, [[10, 12, 11]])
                    np.testing.assert_allclose(distances, [[0, 1, 2]], atol=2e-6, rtol=0)
