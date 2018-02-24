# HNSW - Approximate nearest neighbor search
Paper code for the HNSW 200M SIFT experiment and a header-only C++ HNSW implementation with python bindings.

NEW: Added simple python bindings with incremental construction


#### Test reproduction steps:
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


#### Python bindings example
```python
import hnswlib
import numpy as np

dim = 128
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # Only l2 is supported currently

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
int_labels = p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50)

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

```
To compile run:
```bash
cd python_bindings
python3 setup.py install
```

The repo contrains parts of the Non-Metric Space Library's code https://github.com/searchivarius/nmslib

References:
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv preprint arXiv:1603.09320 (2016).
