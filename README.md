# Hnsw - Approximate nearest neighbor search
Hnsw paper code for the 200M SIFT experiment.

To download and extract the bigann dataset:
```bash
python3 download_bigann.py
```
To compile:
```bash
cmake .
make all
```

To run the test on 200M SIFT subset:
```bash
./main
```

The size of the bigann subset (in millions) is controlled by the variable **subset_size_milllions** hardcoded in **sift_1b.cpp**.

The repo contrains some parts of the Non-Metric Space Library's code https://github.com/searchivarius/nmslib

References:
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv preprint arXiv:1603.09320 (2016).
