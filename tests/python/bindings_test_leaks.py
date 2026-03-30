import gc
import pickle
import unittest

import numpy as np

import hnswlib


class LeakSmokeTestCase(unittest.TestCase):
    def testLeakSmoke(self):
        dim = 8
        max_elements = 200

        for _ in range(25):
            data = np.float32(np.random.random((max_elements, dim)))

            p = hnswlib.Index(space='l2', dim=dim)
            p.init_index(max_elements=max_elements, ef_construction=100, M=16)
            p.add_items(data)

            labels, distances = p.knn_query(data[:25], k=5)
            del labels, distances

            payload = pickle.dumps(p)
            del p
            gc.collect()

            p2 = pickle.loads(payload)
            labels2, distances2 = p2.knn_query(data[:10], k=3)
            del labels2, distances2, p2, payload, data
            gc.collect()
