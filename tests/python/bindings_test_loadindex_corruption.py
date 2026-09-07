import os
import struct
import tempfile
import unittest

import numpy as np

import hnswlib


class LoadIndexCorruptionTestCase(unittest.TestCase):
    def test_malformed_max_elements_throws(self):
        """A crafted index with max_elements_ < cur_element_count must not load."""
        dim = 8
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=2, ef_construction=10, M=4)
        index.add_items(np.float32([[1.0] * dim, [2.0] * dim]))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.bin')
            index.save_index(path)

            # The header starts with offsetLevel0_ (size_t) followed by max_elements_ (size_t).
            # Leave cur_element_count at 2 but shrink max_elements_ to 1 so the allocation
            # would be smaller than the subsequent read.
            size_t_size = struct.calcsize('N')
            with open(path, 'r+b') as f:
                f.seek(size_t_size)
                f.write(struct.pack('N', 1))

            fresh = hnswlib.Index(space='l2', dim=dim)
            with self.assertRaises(RuntimeError):
                fresh.load_index(path)

    def test_legitimate_index_loads(self):
        """Sanity check: a normal saved index still loads and searches correctly."""
        dim = 8
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=10, ef_construction=10, M=4)
        data = np.float32(np.random.random((5, dim)))
        index.add_items(data)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.bin')
            index.save_index(path)

            fresh = hnswlib.Index(space='l2', dim=dim)
            fresh.load_index(path)
            labels, _ = fresh.knn_query(data, k=1)
            self.assertTrue(np.all(labels.reshape(-1) == np.arange(len(data))))


if __name__ == '__main__':
    unittest.main()
