import os
import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        index_classes = (
            hnswlib.Index,
            hnswlib.DoubleIndex,
            hnswlib.Int8Index,
            hnswlib.Int16Index,
            hnswlib.UInt8Index,
            hnswlib.UInt16Index,
        )
        for index_class in index_classes:
            for idx in range(2):
                print("\n**** %s save-load test ****\n" % index_class.__name__)

                np.random.seed(idx)
                dim = 16
                num_elements = 10000

                # Generating sample data
                data = np.random.random((num_elements, dim))
                if np.issubdtype(index_class.data_type, np.integer):
                    # Scale the data to fit the integer bounds:
                    info = np.iinfo(index_class.data_type)
                    data = (data * (info.max - info.min)) + info.min
                data = data.astype(index_class.data_type)

                # Declaring index
                p = index_class(space='l2', dim=dim)  # possible options are l2, cosine or ip

                # Initiating index
                # max_elements - the maximum number of elements, should be known beforehand
                #     (probably will be made optional in the future)
                #
                # ef_construction - controls index search speed/build speed tradeoff
                # M - is tightly connected with internal dimensionality of the data
                #     strongly affects the memory consumption

                p.init_index(max_elements=num_elements, ef_construction=100, M=16)

                # Controlling the recall by setting ef:
                # higher ef leads to better accuracy, but slower search
                p.set_ef(100)

                p.set_num_threads(4)  # by default using all available cores

                # We split the data in two batches:
                data1 = data[:num_elements // 2]
                data2 = data[num_elements // 2:]

                print("Adding first batch of %d elements" % (len(data1)))
                p.add_items(data1)

                # Query the elements for themselves and measure recall:
                labels, distances = p.knn_query(data1)
                distances = distances.reshape(-1)

                # Check that the distances are indeed monotonically increasing:
                is_sorted = np.all(np.diff(distances) >= 0)
                assert is_sorted, ("Expected distances to be sorted in ascending order, but got: " + repr(distances))
                
                items = p.get_items(labels)

                # Check the recall:
                recall = np.mean(labels.reshape(-1) == np.arange(len(data1)))
                self.assertAlmostEqual(recall, 1.0, 3)

                # Check that the returned element data is correct:
                np.testing.assert_allclose(data1, items)

                # Serializing and deleting the index.
                # We need the part to check that serialization is working properly.

                index_path = 'first_half.bin'
                print("Saving index to '%s'" % index_path)
                p.save_index(index_path)
                print("Saved. Deleting...")
                del p
                print("Deleted")

                print("\n**** Mark delete test ****\n")
                # Re-initiating, loading the index
                print("Re-initiating")
                p = index_class(space='l2', dim=dim)

                print("\nLoading index from '%s'\n" % index_path)
                p.load_index(index_path)
                p.set_ef(100)

                print("Adding the second batch of %d elements" % (len(data2)))
                p.add_items(data2)

                # Query the elements for themselves and measure recall:
                labels, distances = p.knn_query(data, k=1)
                items = p.get_items(labels)

                # Check the recall:
                self.assertAlmostEqual(np.mean(labels.reshape(-1) == np.arange(len(data))), 1.0, 3)

                # Check that the returned element data is correct:
                np.testing.assert_allclose(data, items) # deleting index.

                # Checking that all labels are returned correctly:
                sorted_labels = sorted(p.get_ids_list())
                self.assertEqual(np.sum(~np.asarray(sorted_labels) == np.asarray(range(num_elements))), 0)

                # Delete data1
                labels1_deleted, _ = p.knn_query(data1, k=1)

                for l in labels1_deleted:
                    p.mark_deleted(l[0])
                labels2, _ = p.knn_query(data2, k=1)
                items = p.get_items(labels2)
                np.testing.assert_allclose(data2, items)

                labels1_after, _ = p.knn_query(data1, k=1)
                for la in labels1_after:
                    for lb in labels1_deleted:
                        if la[0] == lb[0]:
                            self.assertTrue(False)
                print("All the data in data1 are removed")

                # Checking saving/loading index with elements marked as deleted
                del_index_path = "with_deleted.bin"
                p.save_index(del_index_path)
                p = index_class(space='l2', dim=dim)
                p.load_index(del_index_path)
                p.set_ef(100)

                labels1_after, _ = p.knn_query(data1, k=1)
                for la in labels1_after:
                    for lb in labels1_deleted:
                        if la[0] == lb[0]:
                            self.assertTrue(False)

                # Unmark deleted data
                for l in labels1_deleted:
                    p.unmark_deleted(l[0])
                labels_restored, _ = p.knn_query(data1, k=1)
                self.assertAlmostEqual(np.mean(labels_restored.reshape(-1) == np.arange(len(data1))), 1.0, 3)
                print("All the data in data1 are restored")

            os.remove(index_path)
            os.remove(del_index_path)
