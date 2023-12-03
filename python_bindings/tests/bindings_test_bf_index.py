import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testBFIndex(self):

        dim = 16
        num_elements = 10000
        num_queries = 1000
        k = 20

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        bf_index = hnswlib.BFIndex(space='l2', dim=dim)  # possible options are l2, cosine or ip
        bf_index.init_index(max_elements=num_elements)

        num_threads = 8
        bf_index.set_num_threads(num_threads)  # by default using all available cores

        print(f"Adding all elements {num_elements}")
        bf_index.add_items(data)

        self.assertEqual(bf_index.num_threads, num_threads)
        self.assertEqual(bf_index.get_max_elements(), num_elements)
        self.assertEqual(bf_index.get_current_count(), num_elements)

        queries = np.float32(np.random.random((num_queries, dim)))
        print("Searching nearest neighbours")
        labels, distances = bf_index.knn_query(queries, k=k)

        print("Checking results")
        for i in range(num_queries):
            query = queries[i]
            sq_dists = (data - query)**2
            dists = np.sum(sq_dists, axis=1)
            labels_gt = np.argsort(dists)[:k]
            dists_gt = dists[labels_gt]
            dists_bf = distances[i]
            # we can compare labels but because of numeric errors in distance calculation in C++ and numpy
            # sometimes we get different order of labels, therefore we compare distances
            max_diff_with_gt = np.max(np.abs(dists_gt - dists_bf))

            self.assertTrue(max_diff_with_gt < 1e-5)
