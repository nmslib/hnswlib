import unittest

import numpy as np

import hnswlib


def ordered_clustered_data(num_elements, dim, num_clusters, rng):
    """Points emitted cluster by cluster, so the array is stored in a meaningful
    order (the property a document-ordered corpus has and random data does not)."""
    centers = rng.normal(0, 1.0, (num_clusters, dim)).astype(np.float32)
    per_cluster = num_elements // num_clusters
    return np.concatenate([
        centers[c] + rng.normal(0, 0.35, (per_cluster, dim)).astype(np.float32)
        for c in range(num_clusters)
    ]).astype(np.float32)


def build_and_measure_recall(data, queries, ground_truth, shuffle):
    num_elements, dim = data.shape
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16, random_seed=100)
    p.add_items(data, np.arange(num_elements), shuffle=shuffle)
    p.set_ef(10)
    labels, _ = p.knn_query(queries, k=ground_truth.shape[1])
    k = ground_truth.shape[1]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(labels, ground_truth)])


class ShuffleTestCase(unittest.TestCase):
    def testLabelsAreUnaffectedByShuffling(self):
        """Shuffling reorders the addPoint calls only; the label a vector is
        stored under must not change."""
        dim = 16
        num_elements = 2000
        rng = np.random.default_rng(0)
        data = ordered_clustered_data(num_elements, dim, 20, rng)
        labels = np.arange(num_elements) * 7 + 3

        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=100, M=16)
        p.add_items(data, labels)

        self.assertEqual(sorted(p.get_ids_list()), sorted(labels.tolist()))
        np.testing.assert_allclose(np.float32(p.get_items(labels)), data, rtol=1e-6)

    def testShuffleImprovesRecallOnOrderedData(self):
        """HNSW is analysed as an average case over a random insertion order.
        On a corpus stored in a meaningful order, inserting in that order may cost
        accuracy; shuffling recovers it."""
        dim = 16
        num_elements = 20000
        rng = np.random.default_rng(0)
        data = ordered_clustered_data(num_elements, dim, 100, rng)
        queries = data[rng.choice(num_elements, 500, replace=False)]
        ground_truth = np.argsort(
            ((queries[:, None, :] - data[None, :, :]) ** 2).sum(-1), axis=1)[:, :10]

        recall_in_order = build_and_measure_recall(data, queries, ground_truth, shuffle=False)
        recall_shuffled = build_and_measure_recall(data, queries, ground_truth, shuffle=True)

        print(f"\nordered corpus, recall@10: insertion order {recall_in_order:.4f}, "
              f"shuffled {recall_shuffled:.4f}")
        # The measured gap on this configuration is 0.018-0.025 over repeated
        # builds; assert a fraction of it so the test does not depend on the
        # non-determinism of a multi-threaded build.
        self.assertGreater(recall_shuffled - recall_in_order, 0.005)


if __name__ == "__main__":
    unittest.main()
