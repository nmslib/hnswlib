import os
import pickle
import tempfile
import unittest

import numpy as np

import hnswlib


class SimdDispatchTest(unittest.TestCase):
    def test_get_simd_known(self):
        name = hnswlib.get_simd()
        self.assertIn(name, ("sse", "avx", "avx512", "aarch64"))

    def test_pairwise_l2_finite(self):
        dim = 128
        n = 256
        rng = np.random.RandomState(0)
        data = rng.rand(n, dim).astype(np.float32)
        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(max_elements=n, ef_construction=50, M=8)
        index.add_items(data)
        labels, dist = index.knn_query(data[:8], k=1)
        self.assertEqual(labels.shape, (8, 1))
        self.assertTrue(np.all(np.isfinite(dist)))

    def test_save_load_roundtrip(self):
        dim = 16
        n = 100
        rng = np.random.RandomState(1)
        data = rng.rand(n, dim).astype(np.float32)
        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(max_elements=n, ef_construction=40, M=8)
        index.add_items(data, ids=np.arange(n))
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            index.save_index(path)
            loaded = hnswlib.Index(space="l2", dim=dim)
            loaded.load_index(path)
            self.assertEqual(loaded.get_current_count(), n)
            labels, _ = loaded.knn_query(data[:4], k=1)
            self.assertEqual(labels.shape, (4, 1))
        finally:
            os.remove(path)

    def test_pickle_roundtrip(self):
        dim = 8
        n = 40
        rng = np.random.RandomState(2)
        data = rng.rand(n, dim).astype(np.float32)
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(max_elements=n, ef_construction=40, M=8)
        index.add_items(data)
        blob = pickle.dumps(index)
        restored = pickle.loads(blob)
        self.assertEqual(restored.get_current_count(), n)

    def test_env_sse_override_subprocess(self):
        if hnswlib.get_simd() == "aarch64":
            self.skipTest("x86 override does not apply on aarch64")
        import subprocess
        import sys
        code = "import hnswlib; print(hnswlib.get_simd())"
        env = os.environ.copy()
        env["HNSWLIB_SIMD"] = "sse"
        out = subprocess.check_output([sys.executable, "-c", code], env=env, universal_newlines=True)
        self.assertEqual(out.strip(), "sse")
        env["HNSWLIB_SIMD"] = "not-an-isa"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertNotEqual(proc.returncode, 0)
