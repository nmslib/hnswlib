import os
import pickle
import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        dim = 16
        num_elements = 5000
        max_num_elements = 2 * num_elements

        recall_threshold = 0.98

        # Generating sample data
        print("Generating data")
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
        # batch 4
        first_id += num_elements
        last_id += num_elements
        labels4 = np.arange(first_id, last_id)
        data4 = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        hnsw_index = hnswlib.Index(space='l2', dim=dim)
        hnsw_index.init_index(max_elements=max_num_elements, ef_construction=200, M=16, allow_replace_deleted=True)

        hnsw_index.set_ef(100)
        hnsw_index.set_num_threads(4)

        # Add batch 1 and 2
        print("Adding batch 1")
        hnsw_index.add_items(data1, labels1)
        print("Adding batch 2")
        hnsw_index.add_items(data2, labels2)  # maximum number of elements is reached

        # Delete nearest neighbors of batch 2
        print("Deleting neighbors of batch 2")
        labels2_deleted, _ = hnsw_index.knn_query(data2, k=1)
        # delete probable duplicates from nearest neighbors
        labels2_deleted_no_dup = set(labels2_deleted.flatten())
        num_duplicates = len(labels2_deleted) - len(labels2_deleted_no_dup)
        for l in labels2_deleted_no_dup:
            hnsw_index.mark_deleted(l)
        labels1_found, _ = hnsw_index.knn_query(data1, k=1)
        items = hnsw_index.get_items(labels1_found)
        diff_with_gt_labels = np.mean(np.abs(data1 - items))
        self.assertAlmostEqual(diff_with_gt_labels, 0, delta=1e-3)

        labels2_after, _ = hnsw_index.knn_query(data2, k=1)
        for la in labels2_after:
            if la[0] in labels2_deleted_no_dup:
                print(f"Found deleted label {la[0]} during knn search")
                self.assertTrue(False)
        print("All the neighbors of data2 are removed")

        # Replace deleted elements
        print("Inserting batch 3 by replacing deleted elements")
        # Maximum number of elements is reached therefore we cannot add new items
        # but we can replace the deleted ones
        # Note: there may be less than num_elements elements.
        #       As we could delete less than num_elements because of duplicates
        labels3_tr = labels3[0:labels3.shape[0] - num_duplicates]
        data3_tr = data3[0:data3.shape[0] - num_duplicates]
        hnsw_index.add_items(data3_tr, labels3_tr, replace_deleted=True)

        # After replacing, all labels should be retrievable
        print("Checking that remaining labels are in index")
        # Get remaining data from batch 1 and batch 2 after deletion of elements
        remaining_labels = (set(labels1) | set(labels2)) - labels2_deleted_no_dup
        remaining_labels_list = list(remaining_labels)
        comb_data = np.concatenate((data1, data2), axis=0)
        remaining_data = comb_data[remaining_labels_list]

        returned_items = hnsw_index.get_items(remaining_labels_list)
        self.assertSequenceEqual(remaining_data.tolist(), returned_items)

        returned_items = hnsw_index.get_items(labels3_tr)
        self.assertSequenceEqual(data3_tr.tolist(), returned_items)

        # Check index serialization
        # Delete batch 3
        print("Deleting batch 3")
        for l in labels3_tr:
            hnsw_index.mark_deleted(l)

        # Save index
        index_path = "index.bin"
        print(f"Saving index to {index_path}")
        hnsw_index.save_index(index_path)
        del hnsw_index

        # Reinit and load the index
        hnsw_index = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
        hnsw_index.set_num_threads(4)
        print(f"Loading index from {index_path}")
        hnsw_index.load_index(index_path, max_elements=max_num_elements, allow_replace_deleted=True)

        # Insert batch 4
        print("Inserting batch 4 by replacing deleted elements")
        labels4_tr = labels4[0:labels4.shape[0] - num_duplicates]
        data4_tr = data4[0:data4.shape[0] - num_duplicates]
        hnsw_index.add_items(data4_tr, labels4_tr, replace_deleted=True)

        # Check recall
        print("Checking recall")
        labels_found, _ = hnsw_index.knn_query(data4_tr, k=1)
        recall = np.mean(labels_found.reshape(-1) == labels4_tr)
        print(f"Recall for the 4 batch: {recall}")
        self.assertGreater(recall, recall_threshold)

        # Delete batch 4
        print("Deleting batch 4")
        for l in labels4_tr:
            hnsw_index.mark_deleted(l)

        print("Testing pickle serialization")
        hnsw_index_pckl = pickle.loads(pickle.dumps(hnsw_index))
        del hnsw_index
        # Insert batch 3
        print("Inserting batch 3 by replacing deleted elements")
        hnsw_index_pckl.add_items(data3_tr, labels3_tr, replace_deleted=True)

        # Check recall
        print("Checking recall")
        labels_found, _ = hnsw_index_pckl.knn_query(data3_tr, k=1)
        recall = np.mean(labels_found.reshape(-1) == labels3_tr)
        print(f"Recall for the 3 batch: {recall}")
        self.assertGreater(recall, recall_threshold)

        os.remove(index_path)
