import os
import random
import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):

        dim = 16
        num_elements = 100

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
        bf_index = hnswlib.BFIndex(space='l2', dim=dim)

        # Initiating index
        # max_elements - the maximum number of elements, should be known beforehand
        #     (probably will be made optional in the future)
        #
        # ef_construction - controls index search speed/build speed tradeoff
        # M - is tightly connected with internal dimensionality of the data
        #     strongly affects the memory consumption

        hnsw_index.init_index(max_elements=num_elements, ef_construction=100, M=16)
        bf_index.init_index(max_elements=num_elements)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        hnsw_index.set_ef(10)

        hnsw_index.set_num_threads(4)  # by default using all available cores

        print("Adding %d elements" % (len(data)))
        hnsw_index.add_items(data)
        bf_index.add_items(data)

        # filter_function = lambda id: id%2 == 0
        filter_function = lambda index: index > (num_elements-2)  if random.random() < 0.5 else index > (num_elements-1)
        labels, distances = hnsw_index.knn_query_return_lists(data, k=1, num_threads=1, filter=filter_function)

        # Assert the return type for both labels and distances is a list
        self.assertTrue(isinstance(labels, list))
        self.assertTrue(isinstance(distances, list))
        
        #Check that the length of the returned labels is between 1 and 2
        for label in labels:
            self.assertTrue(len(label) >= 1 and len(label) <= 2)




