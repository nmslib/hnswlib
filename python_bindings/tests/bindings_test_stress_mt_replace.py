import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        dim = 16
        num_elements = 1_000
        max_num_elements = 2 * num_elements

        # Generating sample data
        # batch 1
        first_id = 0
        last_id = num_elements
        labels1 = np.arange(first_id, last_id)
        data1 = np.float32(np.random.random((num_elements, dim)))
        # batch 2
        first_id += num_elements
        last_id += num_elements
        labels2 = np.arange(first_id, last_id)
        data2 = np.float32(np.random.random((num_elements, dim)))
        # batch 3
        first_id += num_elements
        last_id += num_elements
        labels3 = np.arange(first_id, last_id)
        data3 = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        for _ in range(100):
            hnsw_index = hnswlib.Index(space='l2', dim=dim)
            hnsw_index.init_index(max_elements=max_num_elements, ef_construction=200, M=16, replace_deleted=True)

            hnsw_index.set_ef(100)
            hnsw_index.set_num_threads(50)

            # Add batch 1 and 2
            hnsw_index.add_items(data1, labels1)
            hnsw_index.add_items(data2, labels2)  # maximum number of elements is reached

            # Delete nearest neighbors of batch 2
            labels2_deleted, _ = hnsw_index.knn_query(data2, k=1)
            for l in labels2_deleted:
                hnsw_index.mark_deleted(l[0])
            labels1_found, _ = hnsw_index.knn_query(data1, k=1)
            items = hnsw_index.get_items(labels1_found)
            diff_with_gt_labels = np.mean(np.abs(data1-items))
            self.assertAlmostEqual(diff_with_gt_labels, 0, delta=1e-3)

            labels2_after, _ = hnsw_index.knn_query(data2, k=1)
            labels2_after_flat = labels2_after.flatten()
            labels2_deleted_flat = labels2_deleted.flatten()
            common = np.intersect1d(labels2_after_flat, labels2_deleted_flat)
            self.assertTrue(common.size == 0)

            # Replace deleted elements
            # Maximum number of elements is reached therefore we cannot add new items
            # but we can replace the deleted ones
            labels_replaced = hnsw_index.add_items_to_vacant_place(data3, labels3)
       