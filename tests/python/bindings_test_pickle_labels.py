import pickle
import unittest

import numpy as np

import hnswlib


class PickleLabelTests(unittest.TestCase):
    def test_automatic_labels_after_updates_and_copy(self):
        for space in ('l2', 'ip', 'cosine'):
            for copy_method in ('pickle', 'constructor'):
                with self.subTest(space=space, copy_method=copy_method):
                    original = hnswlib.Index(space=space, dim=2)
                    original.init_index(max_elements=8)
                    original.set_num_threads(1)
                    original.add_items(np.array([[0, 1], [1, 0]], dtype=np.float32))
                    original.add_items(np.array([[1, 1]], dtype=np.float32), [0])
                    original.add_items(np.array([[3, 1]], dtype=np.float32))
                    self.assertEqual(sorted(original.get_ids_list()), [0, 1, 3])
                    saved_vector = original.get_items([3]).copy()

                    if copy_method == 'pickle':
                        restored = pickle.loads(pickle.dumps(original))
                    else:
                        restored = hnswlib.Index(original)

                    for index in (original, restored):
                        index.add_items(np.array([[4, 1]], dtype=np.float32))

                    self.assertEqual(restored.get_current_count(), 4)
                    self.assertEqual(sorted(restored.get_ids_list()), [0, 1, 3, 4])
                    np.testing.assert_array_equal(restored.get_items([3]), saved_vector)
                    np.testing.assert_array_equal(
                        restored.get_items([0, 1, 3, 4]),
                        original.get_items([0, 1, 3, 4]),
                    )
                    query = np.array([[1, 0]], dtype=np.float32)
                    expected_labels, expected_distances = original.knn_query(query, k=4)
                    labels, distances = restored.knn_query(query, k=4)
                    self.assertEqual(labels.shape, (1, 4))
                    self.assertEqual(labels.dtype, expected_labels.dtype)
                    self.assertEqual(distances.dtype, np.float32)
                    np.testing.assert_array_equal(labels, expected_labels)
                    np.testing.assert_array_equal(distances, expected_distances)

    def test_legacy_pickle_state_without_label_counter(self):
        original = hnswlib.Index(space='l2', dim=2)
        original.init_index(max_elements=8)
        original.set_num_threads(1)
        original.add_items(np.array([[0, 1], [1, 0]], dtype=np.float32))
        state = dict(original.__getstate__()[0])
        state.pop('cur_l', None)
        restored = hnswlib.Index(state)
        restored.add_items(np.array([[2, 1]], dtype=np.float32))
        self.assertEqual(sorted(restored.get_ids_list()), [0, 1, 2])
        np.testing.assert_array_equal(restored.get_items([2]), [[2, 1]])
