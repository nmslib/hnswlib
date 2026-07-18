"""
Concurrency regression test: load_index()/save_index() must release the GIL.

GitHub issue #1437 (LightspeedDMS/code-indexer): the Index.load_index() and
Index.save_index() pybind11 bindings never released the Python GIL during
the native file read + graph (de)serialization. This blocked ALL Python
threads in the process for the full duration of every HNSW shard load
(multi-hundred-MB/GB files over NFS = seconds to tens of seconds in
production), freezing the code-indexer server's Web UI and MCP front door.

This test builds a real ~300MB on-disk HNSW index, then proves a concurrent
pure-Python "recorder" thread keeps making real wall-clock progress WHILE
load_index() (and, separately, save_index()) runs on the main thread.

Technique: the recorder thread continuously appends time.monotonic()
timestamps to a list while doing a tiny chunk of pure-Python work between
each append. If the GIL is held for the whole native call (unfixed
binding), the recorder thread cannot run AT ALL during that window -- the
largest gap between two consecutive recorded timestamps will be
approximately equal to the call's duration. If the GIL is released (fixed
binding, py::call_guard<py::gil_scoped_release>()), the recorder keeps
appending timestamps every few milliseconds throughout the call, so the
largest gap stays a small fraction of the call duration.

This was validated empirically against the unfixed binding before the fix
was applied: max_gap/call_duration measured ~0.90 for load_index() and
~0.79 for save_index() (i.e. the GIL was held for the vast majority of
each call). An Event.wait()-based rate-comparison harness was tried first
and discarded: it can itself stall waiting to reacquire the GIL from the
very thread it is timing, producing unreliable measurements.
"""
import os
import tempfile
import threading
import time
import unittest

import numpy as np

import hnswlib


class GILReleaseTestCase(unittest.TestCase):
    """Prove load_index()/save_index() release the GIL for the native call."""

    # Tuned (see fork commit message) so save+load each take a meaningful
    # fraction of a second on typical CI/dev hardware -- large enough that
    # recorder-thread starvation cannot be missed, small enough to keep the
    # whole test under ~30s wall clock.
    NUM_ELEMENTS = 500_000
    DIM = 128

    # A stalled (unfixed) binding produces a gap ratio near 0.8-0.9; a fixed
    # binding produces a gap ratio of a few percent. 0.5 sits comfortably
    # between the two, with margin on both sides.
    MAX_GAP_RATIO = 0.5

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        data = np.float32(rng.random((cls.NUM_ELEMENTS, cls.DIM)))

        index = hnswlib.Index(space="l2", dim=cls.DIM)
        index.init_index(max_elements=cls.NUM_ELEMENTS, ef_construction=40, M=8)
        index.set_num_threads(8)
        index.add_items(data)

        fd, cls.index_path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        index.save_index(cls.index_path)
        cls.built_index = index  # reused by the save_index() half of the test

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.index_path):
            os.remove(cls.index_path)

    def _max_recorder_gap(self, blocking_fn, *, min_call_seconds, warmup=0.05, cooldown=0.15):
        """Run `blocking_fn()` on the main thread while a background
        "recorder" thread continuously appends monotonic timestamps.

        Returns (max_gap_seconds, call_duration_seconds).
        """
        stop_flag = threading.Event()
        timestamps = []

        def _recorder():
            while not stop_flag.is_set():
                for _ in range(200):
                    pass
                timestamps.append(time.monotonic())

        thread = threading.Thread(target=_recorder)
        thread.start()
        time.sleep(warmup)  # let the recorder get running before the timed call

        call_start = time.monotonic()
        blocking_fn()
        call_duration = time.monotonic() - call_start

        time.sleep(cooldown)  # gather a few post-call samples too
        stop_flag.set()
        thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "recorder thread never stopped")

        self.assertGreaterEqual(
            call_duration,
            min_call_seconds,
            "native call completed too fast to meaningfully exercise the "
            "GIL-release behavior under test -- increase NUM_ELEMENTS",
        )

        ordered = sorted(timestamps)
        self.assertGreaterEqual(
            len(ordered), 2, "recorder thread produced too few samples to measure a gap"
        )
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        return max(gaps), call_duration

    def test_load_index_releases_gil_during_native_call(self):
        """A concurrent Python recorder thread must keep making real progress
        while load_index() runs on the main thread -- proving the GIL was
        released for the native file-read + deserialize duration."""

        def _do_load():
            fresh_index = hnswlib.Index(space="l2", dim=self.DIM)
            fresh_index.load_index(self.index_path, max_elements=self.NUM_ELEMENTS)

        max_gap, call_duration = self._max_recorder_gap(
            _do_load, min_call_seconds=0.2
        )

        gap_ratio = max_gap / call_duration
        self.assertLess(
            gap_ratio,
            self.MAX_GAP_RATIO,
            f"recorder thread was silent for {gap_ratio:.1%} of load_index()'s "
            f"{call_duration:.3f}s duration (max_gap={max_gap:.3f}s) -- the GIL "
            "was likely held for the whole native call (regression of the "
            "#1437 fix)",
        )

    def test_save_index_releases_gil_during_native_call(self):
        """Same proof as above, for save_index()."""
        save_path = tempfile.mktemp(suffix=".bin")
        try:

            def _do_save():
                self.built_index.save_index(save_path)

            max_gap, call_duration = self._max_recorder_gap(
                _do_save, min_call_seconds=0.05
            )

            gap_ratio = max_gap / call_duration
            self.assertLess(
                gap_ratio,
                self.MAX_GAP_RATIO,
                f"recorder thread was silent for {gap_ratio:.1%} of "
                f"save_index()'s {call_duration:.3f}s duration "
                f"(max_gap={max_gap:.3f}s) -- the GIL was likely held for "
                "the whole native call (regression of the #1437 fix)",
            )
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
