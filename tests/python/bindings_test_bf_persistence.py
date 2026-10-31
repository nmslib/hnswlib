import tempfile
from pathlib import Path
import unittest

import hnswlib
import numpy as np


class BruteForcePersistenceTest(unittest.TestCase):
    def test_update_existing_labels_after_load(self):
        for space in ("l2", "ip", "cosine"):
            for capacity in (2, 3):
                with self.subTest(space=space, capacity=capacity):
                    with tempfile.TemporaryDirectory() as directory:
                        path = str(Path(directory) / "index.bin")
                        index = hnswlib.BFIndex(space=space, dim=2)
                        index.init_index(max_elements=capacity)
                        vectors = np.array([[1, 0], [0, 3]], dtype=np.float32)
                        ids = [101, 4096]
                        index.add_items(vectors, ids)
                        index.save_index(path)
                        loaded = hnswlib.BFIndex(space=space, dim=2)
                        loaded.load_index(path)
                        replacement = np.array([[2, 0]], dtype=np.float32)
                        loaded.add_items(replacement, [101])
                        self.assertEqual(loaded.get_current_count(), 2)
                        labels, distances = loaded.knn_query(replacement, k=2)
                        self.assertEqual(set(labels[0]), set(ids))
                        expected_vectors = {101: replacement[0], 4096: vectors[1]}
                        for label, distance in zip(labels[0], distances[0]):
                            vector = expected_vectors[label]
                            query = replacement[0]
                            if space == "l2":
                                expected = np.sum((vector - query) ** 2)
                            else:
                                if space == "cosine":
                                    vector = vector / np.linalg.norm(vector)
                                    query = query / np.linalg.norm(query)
                                expected = 1 - np.dot(vector, query)
                            self.assertAlmostEqual(
                                float(distance), float(expected), places=5
                            )
                        loaded.save_index(path)
                        again = hnswlib.BFIndex(space=space, dim=2)
                        again.load_index(path)
                        again.add_items(vectors[1:], [4096])
                        self.assertEqual(again.get_current_count(), 2)


if __name__ == "__main__":
    unittest.main()
