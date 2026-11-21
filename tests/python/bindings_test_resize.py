import unittest

import numpy as np

import hnswlib


class BruteForceResizeTestCase(unittest.TestCase):
    def test_resize(self):
        index = hnswlib.BFIndex(space="l2", dim=2)
        index.init_index(max_elements=2)
        vectors = np.float32([[1, 2], [3, 4], [5, 6]])
        index.add_items(vectors[:2], [11, 22])

        index.resize_index(new_size=4)
        index.add_items(vectors[2:], [33])
        index.resize_index(3)
        with self.assertRaises(RuntimeError):
            index.resize_index(2)
        with self.assertRaises(RuntimeError):
            index.resize_index(np.iinfo(np.uintp).max)

        vectors[0] = [7, 8]
        index.add_items(vectors[:1], [11])
        self.assertEqual(index.get_max_elements(), 3)
        self.assertEqual(index.get_current_count(), 3)
        labels, distances = index.knn_query(vectors, k=1)
        np.testing.assert_array_equal(labels[:, 0], [11, 22, 33])
        np.testing.assert_array_equal(distances[:, 0], [0, 0, 0])

    def test_resize_empty(self):
        index = hnswlib.BFIndex(space="l2", dim=2)
        with self.assertRaises(RuntimeError):
            index.resize_index(1)
        index.init_index(max_elements=2)
        index.resize_index(0)
        self.assertEqual(index.get_max_elements(), 0)
        index.resize_index(1)
        index.add_items([[1, 2]])
        self.assertEqual(index.get_current_count(), 1)


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        for idx in range(16):
            print("\n**** Index resize test ****\n")

            np.random.seed(idx)
            dim = 16
            num_elements = 10000

            # Generating sample data
            data = np.float32(np.random.random((num_elements, dim)))

            # Declaring index
            p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

            # Initiating index
            # max_elements - the maximum number of elements, should be known beforehand
            #     (probably will be made optional in the future)
            #
            # ef_construction - controls index search speed/build speed tradeoff
            # M - is tightly connected with internal dimensionality of the data
            #     strongly affects the memory consumption

            p.init_index(max_elements=num_elements//2, ef_construction=100, M=16)

            # Controlling the recall by setting ef:
            # higher ef leads to better accuracy, but slower search
            p.set_ef(20)

            p.set_num_threads(idx % 8)  # by default using all available cores

            # We split the data in two batches:
            data1 = data[:num_elements // 2]
            data2 = data[num_elements // 2:]

            print("Adding first batch of %d elements" % (len(data1)))
            p.add_items(data1)

            # Query the elements for themselves and measure recall:
            labels, distances = p.knn_query(data1, k=1)

            items = p.get_items(list(range(len(data1))))

            # Check the recall:
            self.assertAlmostEqual(np.mean(labels.reshape(-1) == np.arange(len(data1))), 1.0, 3)

            # Check that the returned element data is correct:
            diff_with_gt_labels = np.max(np.abs(data1-items))
            self.assertAlmostEqual(diff_with_gt_labels, 0, delta=1e-4)

            print("Resizing the index")
            p.resize_index(num_elements)

            print("Adding the second batch of %d elements" % (len(data2)))
            p.add_items(data2)

            # Query the elements for themselves and measure recall:
            labels, distances = p.knn_query(data, k=1)
            items=p.get_items(list(range(num_elements)))

            # Check the recall:
            self.assertAlmostEqual(np.mean(labels.reshape(-1) == np.arange(len(data))), 1.0, 3)

            # Check that the returned element data is correct:
            diff_with_gt_labels = np.max(np.abs(data-items))
            self.assertAlmostEqual(diff_with_gt_labels, 0, delta=1e-4)

            # Checking that all labels are returned correctly:
            sorted_labels = sorted(p.get_ids_list())
            self.assertEqual(np.sum(~np.asarray(sorted_labels) == np.asarray(range(num_elements))), 0)
