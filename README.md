# HNSW - Approximate nearest neighbor search
Header-only C++ HNSW implementation with python bindings. Paper code for the HNSW 200M SIFT experiment

NEW: Added support for cosine similarity and inner product distances


Part of the nmslib project https://github.com/nmslib/nmslib



Supported distances:
1) Squared L2 ('l2')
2) Inner product ('ip',the distance is 1.0 - $inner product$)
3) Cosine similarity ('cosine', the same as the inner product, but vectors are normalized)

For other spaces use the main library https://github.com/nmslib/nmslib 


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
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
int_labels = p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

```
To compile run:
```bash
cd python_bindings
python3 setup.py install
```

#### 200M SIFT test reproduction steps:
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



References:
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv preprint arXiv:1603.09320 (2016).
