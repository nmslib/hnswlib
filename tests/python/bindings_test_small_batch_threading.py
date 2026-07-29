import multiprocessing
import statistics
import time
import unittest

import numpy as np

import hnswlib


class SmallBatchHighDimThreadingTestCase(unittest.TestCase):
    """Regression test for https://github.com/nmslib/hnswlib/issues/667.

    With the old rows-only threshold, knn_query forced single-threaded
    execution for small batches. For high-dimensional vectors this hurts
    performance because each vector still does enough work to benefit from
    parallelization.
    """

    def setUp(self):
        self.num_elements = 20000
        self.dim = 1024
        self.batch_size = 8
        self.k = 10
        np.random.seed(42)
        self.data = np.float32(np.random.random((self.num_elements, self.dim)))
        self.queries = np.float32(np.random.random((self.batch_size, self.dim)))

        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(max_elements=self.num_elements, ef_construction=200, M=16)
        # Build the index with all available cores; this path is unaffected by
        # the query-time threading heuristic.
        self.index.add_items(self.data, num_threads=-1)
        self.index.set_ef(200)

    def _median_query_time(self, num_threads, repeats=10, loops=10):
        self.index.set_num_threads(num_threads)
        # Warmup to ensure any lazy state is created before timing.
        self.index.knn_query(self.queries, k=self.k)

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(loops):
                self.index.knn_query(self.queries, k=self.k)
            times.append(time.perf_counter() - start)
        return statistics.median(times)

    def test_small_batch_high_dim_query_uses_multiple_threads(self):
        # Skip on single-core machines where parallelization cannot help.
        if multiprocessing.cpu_count() < 2:
            self.skipTest('Need at least 2 CPUs to measure parallel speedup')

        t_single = self._median_query_time(num_threads=1)
        t_multi = self._median_query_time(num_threads=4)

        # Correctness must be identical regardless of thread count.
        self.index.set_num_threads(1)
        labels_single, distances_single = self.index.knn_query(self.queries, k=self.k)
        self.index.set_num_threads(4)
        labels_multi, distances_multi = self.index.knn_query(self.queries, k=self.k)

        np.testing.assert_array_equal(labels_single, labels_multi)
        np.testing.assert_allclose(distances_single, distances_multi, rtol=1e-5)

        # The bug manifested as t_multi ~= t_single because the small batch
        # forced serial execution. After the fix, multiple threads should be
        # measurably faster. Guard against noise with a conservative threshold.
        speedup = t_single / t_multi
        self.assertGreater(
            speedup, 1.3,
            f"Expected parallel query to be faster for high-dim small batch, "
            f"got speedup {speedup:.2f}x (single={t_single:.4f}s, "
            f"multi={t_multi:.4f}s)"
        )
