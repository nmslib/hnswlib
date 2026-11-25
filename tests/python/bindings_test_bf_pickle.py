import pickle
import unittest

import numpy as np

import hnswlib


class BFPickleUnitTests(unittest.TestCase):
    def test_uninitialized(self):
        index = hnswlib.BFIndex('l2', 3)
        index.num_threads = 2
        restored = pickle.loads(pickle.dumps(index))
        self.assertEqual(repr(restored), repr(index))
        self.assertEqual(restored.num_threads, 2)
        restored.init_index(4)
        restored.add_items([[1, 2, 3]])
        self.assertEqual(restored.knn_query([[1, 2, 3]])[0][0, 0], 0)

    def test_empty(self):
        index = hnswlib.BFIndex('l2', 3)
        index.init_index(4)
        restored = pickle.loads(pickle.dumps(index))
        self.assertEqual(restored.get_current_count(), 0)
        self.assertEqual(restored.get_max_elements(), 4)
        restored.add_items([[1, 2, 3]])
        self.assertEqual(restored.knn_query([[1, 2, 3]])[0][0, 0], 0)

    def test_search_and_updates(self):
        data = np.array([[1, 2, 3], [4, 1, 2], [2, 5, 1]], dtype=np.float32)
        queries = np.array([[2, 3, 1], [1, 4, 5]], dtype=np.float32)
        for space in ('l2', 'ip', 'cosine'):
            for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(space=space, protocol=protocol):
                    index = hnswlib.BFIndex(space, 3)
                    index.init_index(8)
                    index.num_threads = 2
                    index.add_items(data, [91, 7, 400])
                    expected = index.knn_query(queries, k=3)
                    restored = pickle.loads(pickle.dumps(index, protocol=protocol))
                    self.assertEqual(restored.get_current_count(), 3)
                    self.assertEqual(restored.get_max_elements(), 8)
                    self.assertEqual(restored.num_threads, 2)
                    actual = restored.knn_query(queries, k=3)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    # Restoring labels must also rebuild the lookup used for updates.
                    replacement = [[9, 8, 7]]
                    restored.add_items(replacement, [91])
                    index.add_items(replacement, [91])
                    self.assertEqual(restored.get_current_count(), 3)
                    # The next automatic label counts inserted rows, including updates.
                    restored.add_items([[7, 8, 9]])
                    index.add_items([[7, 8, 9]])
                    self.assertEqual(restored.get_current_count(), 4)
                    actual = restored.knn_query(queries, k=4)
                    expected = index.knn_query(queries, k=4)
                    self.assertIn(4, actual[0][0])
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    # Objects own separate storage after the round-trip.
                    self.assertEqual(index.get_current_count(), 4)
                    restored.add_items([[2, 2, 2]], [99])
                    self.assertEqual(index.get_current_count(), 4)

    def test_automatic_label_counter_after_updates(self):
        index = hnswlib.BFIndex('l2', 2)
        index.init_index(5)
        index.add_items([[1, 2]], [50])
        index.add_items([[3, 4]], [50])
        restored = pickle.loads(pickle.dumps(index))
        restored.add_items([[9, 9]])
        self.assertEqual(restored.get_current_count(), 2)
        self.assertEqual(restored.knn_query([[9, 9]])[0][0, 0], 2)

    def test_unused_capacity_is_not_serialized(self):
        sizes = []
        for capacity in (1, 10000):
            index = hnswlib.BFIndex('l2', 3)
            index.init_index(capacity)
            index.add_items([[1, 2, 3]], [42])
            sizes.append(len(pickle.dumps(index)))
        # Only the capacity metadata should grow, not the unused row storage.
        self.assertLess(abs(sizes[1] - sizes[0]), 16)

    def test_invalid_state(self):
        index = hnswlib.BFIndex('l2', 3)
        index.init_index(5)
        index.add_items([[1, 2, 3]], [42])
        state = index.__getstate__()[0]
        invalid = [
            dict(state, ser_version=999),
            dict(state, dim=0),
            dict(state, cur_element_count=6),
            dict(state, data=state['data'][:-1]),
            dict(state, data=state['data'] + b'X'),
            dict(state, max_elements=np.iinfo(np.uintp).max),
            dict(state, cur_element_count=2, data=state['data'] * 2),
        ]
        for params in invalid:
            with self.subTest(params=params):
                restored = hnswlib.BFIndex.__new__(hnswlib.BFIndex)
                with self.assertRaisesRegex(RuntimeError, 'Unpickle Error:'):
                    restored.__setstate__((params,))
        restored = hnswlib.BFIndex.__new__(hnswlib.BFIndex)
        with self.assertRaisesRegex(RuntimeError, 'Invalid state'):
            restored.__setstate__((state, state))


if __name__ == '__main__':
    unittest.main()
