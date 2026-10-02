import unittest

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


class CosineNormalizationTestCase(unittest.TestCase):
    def _check_distances(self, index_type, data, queries, expected):
        index = index_type(space='cosine', dim=data.shape[1])
        if index_type is hnswlib.Index:
            index.init_index(max_elements=len(data), random_seed=100)
            index.set_ef(10)
        else:
            index.init_index(max_elements=len(data))
        index.set_num_threads(1)
        index.add_items(data, np.arange(len(data)))
        labels, distances = index.knn_query(
            queries, k=len(data), num_threads=1)

        # Every point is requested; compare distances by ID, including ties.
        np.testing.assert_array_equal(
            np.sort(labels, axis=1),
            np.broadcast_to(np.arange(len(data)), labels.shape))
        reference_by_id = expected[np.arange(len(queries))[:, None], labels]
        np.testing.assert_allclose(
            distances, reference_by_id, rtol=0, atol=2e-6)

    def test_small_nonzero_vectors(self):
        # These literal distances follow from identical/opposite directions.
        expected = np.array([[0.0, 2.0], [2.0, 0.0]])
        for index_type in (hnswlib.Index, hnswlib.BFIndex):
            for dim in (3, 8, 16):
                for magnitude in (1.0, 1e-12, 1e-25):
                    with self.subTest(index=index_type.__name__, dim=dim,
                                      magnitude=magnitude):
                        data = np.zeros((2, dim), dtype=np.float32)
                        data[:, 0] = [magnitude, -magnitude]
                        self._check_distances(
                            index_type, data, data, expected)

    def test_cosine_distances_across_stored_and_query_scales(self):
        smallest = np.nextafter(np.float32(0), np.float32(1))
        scales = (
            (1.0, 1.0),
            (1e-25, 1.0),
            (1.0, 1e-25),
            (1e-25, 1e-25),
            (1e-40, 1.0),
            (1.0, 1e-40),
            (1e-40, 1e-40),
            (smallest, smallest),
            (1e30, 1.0),
            (1.0, 1e30),
            (1e30, 1e30),
        )
        for index_type in (hnswlib.Index, hnswlib.BFIndex):
            for dim in (3, 8, 16):
                directions = np.zeros((4, dim), dtype=np.float32)
                directions[:, :2] = [[1, 0], [3, 4], [-1, 0], [0, 1]]
                for stored_scale, query_scale in scales:
                    with self.subTest(index=index_type.__name__, dim=dim,
                                      stored_scale=stored_scale,
                                      query_scale=query_scale):
                        data = directions * np.float32(stored_scale)
                        queries = directions[:2] * np.float32(query_scale)

                        # Independent cosine formula on the actual float32
                        # inputs, promoted before squaring or dot products.
                        wide_data = data.astype(np.float64)
                        wide_queries = queries.astype(np.float64)
                        dot = wide_queries @ wide_data.T
                        lengths = (
                            np.linalg.norm(wide_queries, axis=1)[:, None]
                            * np.linalg.norm(wide_data, axis=1)[None, :])
                        expected = 1.0 - dot / lengths
                        self._check_distances(
                            index_type, data, queries, expected)

    def test_cosine_distances_near_float_norm_limits(self):
        limits = np.finfo(np.float32)
        root_min = np.float32(np.sqrt(float(limits.tiny)))
        for index_type in (hnswlib.Index, hnswlib.BFIndex):
            for dim in (3, 8, 16, 128):
                root_max = np.float32(np.sqrt(float(limits.max) / dim))
                scales = (
                    np.float32(1e-22),
                    np.float32(1e-21),
                    root_min / np.float32(2),
                    np.nextafter(root_min, np.float32(0)),
                    root_min,
                    np.nextafter(root_min, np.float32(np.inf)),
                    np.nextafter(root_max, np.float32(0)),
                    root_max,
                    np.nextafter(root_max, np.float32(np.inf)),
                    np.float32(limits.max / np.float32(4)),
                )
                # Dense directions exercise rounding across many components.
                directions = np.ones((4, dim), dtype=np.float32)
                directions[1, ::2] = 3
                directions[1, 1::2] = 4
                directions[2] = -1
                directions[3, 1::2] = 0
                for scale in scales:
                    for stored_scale, query_scale in (
                            (scale, 1.0), (1.0, scale), (scale, scale)):
                        with self.subTest(index=index_type.__name__, dim=dim,
                                          stored_scale=stored_scale,
                                          query_scale=query_scale):
                            data = directions * np.float32(stored_scale)
                            queries = directions[:2] * np.float32(query_scale)
                            wide_data = data.astype(np.float64)
                            wide_queries = queries.astype(np.float64)
                            dot = wide_queries @ wide_data.T
                            lengths = (
                                np.linalg.norm(wide_queries, axis=1)[:, None]
                                * np.linalg.norm(wide_data, axis=1)[None, :])
                            expected = 1.0 - dot / lengths
                            self._check_distances(
                                index_type, data, queries, expected)

    def test_zero_vector_compatibility(self):
        # Existing convention: zero has dot product zero with every vector.
        expected = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ])
        for index_type in (hnswlib.Index, hnswlib.BFIndex):
            data = np.zeros((3, 8), dtype=np.float32)
            data[1, 0] = 1.0
            data[2, 0] = -1.0
            with self.subTest(index=index_type.__name__):
                self._check_distances(index_type, data, data, expected)
